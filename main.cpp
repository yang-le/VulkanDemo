/*
 * Vulkan Windowed Program
 *
 * Copyright (C) 2016, 2018 Valve Corporation
 * Copyright (C) 2016, 2018 LunarG, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
Vulkan C++ Windowed Project Template
Create and destroy a Vulkan surface on an SDL window.
*/

// Enable the WSI extensions
#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

// Tell SDL not to mess with main()
#define SDL_MAIN_HANDLED

#include <glm/glm.hpp>
#include <SDL2/SDL.h>
#include <SDL2/SDL_syswm.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.hpp>

#include <iostream>
#include <vector>
#include <optional>
#include <set>
#include <fstream>

class HelloVulkan {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanUp();
    }

private:
    const int MAX_FRAMES_IN_FLIGHT = 2;
    size_t currentFrame = 0;

    SDL_Window* window = nullptr;
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;
    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    std::vector<vk::Framebuffer> swapChainFrambuffers;
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishSemaphores;
    std::vector<vk::Fence> inFlightFences;

    std::vector<const char*> layers;
    const std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    void initWindow() {
        // Create an SDL window that supports Vulkan rendering.
        if(SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::cout << "Could not initialize SDL." << std::endl;
        }
        window = SDL_CreateWindow("Vulkan Window", SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_VULKAN);
        if(window == NULL) {
            std::cout << "Could not create SDL window." << std::endl;
        }   
    }

    void createInstance() {
        // Get WSI extensions from SDL (we can add more if we like - we just can't remove these)
        unsigned extension_count;
        if(!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, NULL)) {
            std::cout << "Could not get the number of required instance extensions from SDL." << std::endl;
        }
        std::vector<const char*> extensions(extension_count);
        if(!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data())) {
            std::cout << "Could not get the names of required instance extensions from SDL." << std::endl;
        }

         // Use validation layers if this is a debug build
    #if defined(_DEBUG)
        layers.emplace_back("VK_LAYER_KHRONOS_validation");
        extensions.emplace_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    #endif

        // vk::ApplicationInfo allows the programmer to specifiy some basic information about the
        // program, which can be useful for layers and tools to provide more debug information.
        vk::ApplicationInfo appInfo = vk::ApplicationInfo()
            .setPApplicationName("Vulkan C++ Windowed Program Template")
            .setApplicationVersion(1)
            .setPEngineName("LunarG SDK")
            .setEngineVersion(1)
            .setApiVersion(VK_API_VERSION_1_0);

        // vk::InstanceCreateInfo is where the programmer specifies the layers and/or extensions that
        // are needed.
        vk::InstanceCreateInfo instInfo = vk::InstanceCreateInfo()
            .setFlags(vk::InstanceCreateFlags())
            .setPApplicationInfo(&appInfo)
            .setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()))
            .setPpEnabledExtensionNames(extensions.data())
            .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
            .setPpEnabledLayerNames(layers.data());

        // Create the Vulkan instance.
        try {
            instance = vk::createInstance(instInfo);
        } catch(const std::exception& e) {
            std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        }
    }

    void createSurface() {
        // Create a Vulkan surface for rendering
        VkSurfaceKHR c_surface;
        if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(instance), &c_surface)) {
            std::cout << "Could not create a Vulkan surface." << std::endl;
            return;
        }
        surface = vk::SurfaceKHR(c_surface);
    }

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    QueueFamilyIndices findQueueFamilies(const vk::PhysicalDevice& device) {
        QueueFamilyIndices indices;

        int i = 0;
        for (const auto& queue : device.getQueueFamilyProperties()) {
            if (queue.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            if (device.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }
            i++;
        }

        return indices;
    }

    struct SwapChainSupportDetails {
        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    SwapChainSupportDetails querySwapChainSupport(const vk::PhysicalDevice& device) {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
        details.formats = device.getSurfaceFormatsKHR(surface);
        details.presentModes = device.getSurfacePresentModesKHR(surface);

        return details;
    }

    bool checkDeviceExtensionSupport(const vk::PhysicalDevice& device) {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool isDeviceSuitable(const vk::PhysicalDevice& device) {
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionsSupported = checkDeviceExtensionSupport(device);
        bool swapChainAdequate = false;
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    void pickPhysicalDevice() {
        for (const auto& device : instance.enumeratePhysicalDevices()) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
            }
        }
    }

    void createLogicalDevice() {
        float queuePriority = 1.0f;
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<unsigned int> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        for (unsigned int queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo = vk::DeviceQueueCreateInfo()
                .setQueueFamilyIndex(indices.graphicsFamily.value())
                .setQueueCount(1)
                .setPQueuePriorities(&queuePriority);
            queueCreateInfos.emplace_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures = vk::PhysicalDeviceFeatures();
        vk::DeviceCreateInfo createInfo = vk::DeviceCreateInfo()
            .setPQueueCreateInfos(queueCreateInfos.data())
            .setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()))
            .setPEnabledFeatures(&deviceFeatures)
            .setEnabledLayerCount(static_cast<uint32_t>(layers.size()))
            .setPpEnabledLayerNames(layers.data())
            .setEnabledExtensionCount(static_cast<uint32_t>(deviceExtensions.size()))
            .setPpEnabledExtensionNames(deviceExtensions.data());
        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        if (availableFormats.size() == 1 && availableFormats[0].format == vk::Format::eUndefined) {
            return { vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear };
        }

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox || availablePresentMode == vk::PresentModeKHR::eImmediate) {
                return availablePresentMode;
            }
        }

        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        int width, height;
        SDL_Vulkan_GetDrawableSize(window, &width, &height);
        
        vk::Extent2D actualExtent;
        actualExtent.width = std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
        
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR createInfo = vk::SwapchainCreateInfoKHR()
            .setSurface(surface)
            .setMinImageCount(imageCount)
            .setImageFormat(surfaceFormat.format)
            .setImageColorSpace(surfaceFormat.colorSpace)
            .setImageExtent(extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setPreTransform(swapChainSupport.capabilities.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(presentMode)
            .setClipped(VK_TRUE)
            .setOldSwapchain(nullptr);

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo
                .setImageSharingMode(vk::SharingMode::eConcurrent)
                .setQueueFamilyIndexCount(2)
                .setPQueueFamilyIndices(queueFamilyIndices);
        } else {
            createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        }

        swapChain = device.createSwapchainKHR(createInfo);
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        vk::ImageViewCreateInfo createInfo = vk::ImageViewCreateInfo()
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swapChainImageFormat)
            .setComponents(vk::ComponentMapping())
            .setSubresourceRange(vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

        for (const auto& swapChainImage : swapChainImages) {
            createInfo.setImage(swapChainImage);
            swapChainImageViews.emplace_back(device.createImageView(createInfo));
        }
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment = vk::AttachmentDescription()
            .setFormat(swapChainImageFormat)
            .setSamples(vk::SampleCountFlagBits::e1)
            .setLoadOp(vk::AttachmentLoadOp::eClear)
            .setStoreOp(vk::AttachmentStoreOp::eStore)
            .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
            .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
            .setInitialLayout(vk::ImageLayout::eUndefined)
            .setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

        vk::AttachmentReference colorAttachmentRef = vk::AttachmentReference()
            .setAttachment(0)
            .setLayout(vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass = vk::SubpassDescription()
            .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
            .setColorAttachmentCount(1)
            .setPColorAttachments(&colorAttachmentRef);

        vk::SubpassDependency dependency = vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlagBits::eNoneKHR)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo = vk::RenderPassCreateInfo()
            .setAttachmentCount(1)
            .setPAttachments(&colorAttachment)
            .setSubpassCount(1)
            .setPSubpasses(&subpass)
            .setDependencyCount(1)
            .setPDependencies(&dependency);

        renderPass = device.createRenderPass(renderPassInfo);
    }

    std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);

        file.seekg(0);
        file.read(buffer.data(), fileSize);

        file.close();

        return buffer;
    }

    vk::ShaderModule createShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
            .setCodeSize(code.size())
            .setPCode(reinterpret_cast<const uint32_t*>(code.data()));

        return device.createShaderModule(createInfo);
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("vert.spv");
        auto fragShaderCode = readFile("frag.spv");

        vk::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(vertShaderModule)
            .setPName("main");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(fragShaderModule)
            .setPName("main");

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = vk::PipelineVertexInputStateCreateInfo()
            .setVertexBindingDescriptionCount(0)
            .setVertexAttributeDescriptionCount(0);

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = vk::PipelineInputAssemblyStateCreateInfo()
            .setTopology(vk::PrimitiveTopology::eTriangleList)
            .setPrimitiveRestartEnable(VK_FALSE);

        vk::Viewport viewport = vk::Viewport()
            .setX(0)
            .setY(0)
            .setWidth((float)swapChainExtent.width)
            .setHeight((float)swapChainExtent.height)
            .setMinDepth(0.0f)
            .setMaxDepth(1.0f);

        vk::Rect2D scissor = vk::Rect2D()
            .setOffset({ 0, 0 })
            .setExtent(swapChainExtent);

        vk::PipelineViewportStateCreateInfo viewportState = vk::PipelineViewportStateCreateInfo()
            .setViewportCount(1)
            .setPViewports(&viewport)
            .setScissorCount(1)
            .setPScissors(&scissor);

        vk::PipelineRasterizationStateCreateInfo rasterizer = vk::PipelineRasterizationStateCreateInfo()
            .setDepthClampEnable(VK_FALSE)
            .setRasterizerDiscardEnable(VK_FALSE)
            .setLineWidth(1.0f)
            .setCullMode(vk::CullModeFlagBits::eBack)
            .setFrontFace(vk::FrontFace::eClockwise)
            .setDepthBiasEnable(VK_FALSE);

        vk::PipelineMultisampleStateCreateInfo multisampling = vk::PipelineMultisampleStateCreateInfo()
            .setSampleShadingEnable(VK_FALSE)
            .setRasterizationSamples(vk::SampleCountFlagBits::e1)
            .setMinSampleShading(1.0f);

        vk::PipelineColorBlendAttachmentState colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
            .setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA)
            .setBlendEnable(VK_FALSE);

        vk::PipelineColorBlendStateCreateInfo colorBlending = vk::PipelineColorBlendStateCreateInfo()
            .setLogicOpEnable(VK_FALSE)
            .setAttachmentCount(1)
            .setPAttachments(&colorBlendAttachment);

        vk::DynamicState dynamicStates[] = { vk::DynamicState::eViewport, vk::DynamicState::eLineWidth };
        vk::PipelineDynamicStateCreateInfo dynamicState = vk::PipelineDynamicStateCreateInfo()
            .setDynamicStateCount(2)
            .setPDynamicStates(dynamicStates);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
            .setStageCount(2)
            .setPStages(shaderStages)
            .setPVertexInputState(&vertexInputInfo)
            .setPInputAssemblyState(&inputAssembly)
            .setPViewportState(&viewportState)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampling)
            .setPColorBlendState(&colorBlending)
            .setLayout(pipelineLayout)
            .setRenderPass(renderPass)
            .setSubpass(0);

        vk::PipelineCache pipelineCache = vk::PipelineCache();
        
        graphicsPipeline = device.createGraphicsPipeline(pipelineCache, pipelineInfo).value;

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
    }

    void createFramebuffers() {
        for (const auto& swapChainImageView : swapChainImageViews) {
            vk::FramebufferCreateInfo framebufferInfo = vk::FramebufferCreateInfo()
                .setRenderPass(renderPass)
                .setAttachmentCount(1)
                .setPAttachments(&swapChainImageView)
                .setWidth(swapChainExtent.width)
                .setHeight(swapChainExtent.height)
                .setLayers(1);

            swapChainFrambuffers.emplace_back(device.createFramebuffer(framebufferInfo));
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo poolInfo = vk::CommandPoolCreateInfo()
            .setQueueFamilyIndex(queueFamilyIndices.graphicsFamily.value());

        commandPool = device.createCommandPool(poolInfo);
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocInfo = vk::CommandBufferAllocateInfo()
            .setCommandPool(commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(swapChainFrambuffers.size());

        commandBuffers = device.allocateCommandBuffers(allocInfo);

        vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        
        vk::ClearValue clearColor = vk::ClearValue()
            .setColor(std::array{ 0.0f, 0.0f, 0.0f, 1.0f });

        for (size_t i = 0; i < commandBuffers.size(); ++i) {
            commandBuffers[i].begin(beginInfo);

            vk::RenderPassBeginInfo renderPassInfo = vk::RenderPassBeginInfo()
                .setRenderPass(renderPass)
                .setFramebuffer(swapChainFrambuffers[i])
                .setRenderArea({ {0, 0}, swapChainExtent })
                .setClearValueCount(1)
                .setPClearValues(&clearColor);

            commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
            commandBuffers[i].draw(3, 1, 0, 0);
            commandBuffers[i].endRenderPass();

            commandBuffers[i].end();
        }
    }

    void createSyncObjects() {
        vk::SemaphoreCreateInfo semaphoreInfo = vk::SemaphoreCreateInfo();
        vk::FenceCreateInfo fenceInfo = vk::FenceCreateInfo()
            .setFlags(vk::FenceCreateFlagBits::eSignaled);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            imageAvailableSemaphores.emplace_back(device.createSemaphore(semaphoreInfo));
            renderFinishSemaphores.emplace_back(device.createSemaphore(semaphoreInfo));
            inFlightFences.emplace_back(device.createFence(fenceInfo));
        }
    }

    void initVulkan() {
        createInstance();
    #if defined(_DEBUG)
        setupDebugCallback();
    #endif
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSyncObjects();
    }

    void drawFrame() {
        device.waitForFences(1, &inFlightFences[currentFrame], VK_TRUE, std::numeric_limits<uint64_t>::max());
        device.resetFences(1, &inFlightFences[currentFrame]);

        uint32_t imageIndex = device.acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrame]);
        
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo = vk::SubmitInfo()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&imageAvailableSemaphores[currentFrame])
            .setPWaitDstStageMask(&waitStage)
            .setCommandBufferCount(1)
            .setPCommandBuffers(&commandBuffers[imageIndex])
            .setSignalSemaphoreCount(1)
            .setPSignalSemaphores(&renderFinishSemaphores[currentFrame]);

        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

        vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
            .setWaitSemaphoreCount(1)
            .setPWaitSemaphores(&renderFinishSemaphores[currentFrame])
            .setSwapchainCount(1)
            .setPSwapchains(&swapChain)
            .setPImageIndices(&imageIndex);

        presentQueue.presentKHR(presentInfo);
        
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop() {
        // Poll for user input.
        bool stillRunning = true;
        while(stillRunning) {
            SDL_Event event;
            while(SDL_PollEvent(&event)) {
                switch(event.type) {
                case SDL_QUIT:
                    stillRunning = false;
                    break;
                default:
                    // Do nothing.
                    break;
                }
            }
        
            drawFrame();

            SDL_Delay(10);
        }

        device.waitIdle();
    }

    void cleanUp() {
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            device.destroyFence(inFlightFences[i]);
            device.destroySemaphore(renderFinishSemaphores[i]);
            device.destroySemaphore(imageAvailableSemaphores[i]);
        }
        device.destroyCommandPool(commandPool);
        for (const auto& framebuffer : swapChainFrambuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
        for (const auto& imageView : swapChainImageViews) {
            device.destroyImageView(imageView);
        }
        device.destroySwapchainKHR(swapChain);
        device.destroy();
    #if defined(_DEBUG)
        instance.destroyDebugUtilsMessengerEXT(debugMessager, nullptr, vk::DispatchLoaderDynamic(static_cast<VkInstance>(instance), vkGetInstanceProcAddr));
    #endif
        instance.destroySurfaceKHR(surface);
        instance.destroy();

        SDL_DestroyWindow(window);
        SDL_Quit();
    }

#if defined(_DEBUG)
    void setupDebugCallback() {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo = vk::DebugUtilsMessengerCreateInfoEXT()
            .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo /*| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose*/)
            .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
            .setPfnUserCallback(debugCallback);
        debugMessager = instance.createDebugUtilsMessengerEXT(debugInfo, nullptr, vk::DispatchLoaderDynamic(static_cast<VkInstance>(instance), vkGetInstanceProcAddr));
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT          messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT                 messageTypes,
        const VkDebugUtilsMessengerCallbackDataEXT*     pCallbackData,
        void* pUserData
    ) {
        std::cout
            << vk::to_string(vk::DebugUtilsMessageSeverityFlagsEXT(messageSeverity))
            << vk::to_string(vk::DebugUtilsMessageTypeFlagsEXT(messageTypes))
            << pCallbackData->pMessage << std::endl;
        return VK_FALSE;
    }

    vk::DebugUtilsMessengerEXT debugMessager;
#endif
};

int main() {
    HelloVulkan app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}