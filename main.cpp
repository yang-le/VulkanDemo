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
#include <vulkan/vulkan_raii.hpp>

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

    vk::raii::Context context;

    SDL_Window* window = nullptr;
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::raii::SwapchainKHR swapChain = nullptr;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;
    vk::raii::RenderPass renderPass = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    std::vector<vk::raii::Framebuffer> swapChainFrambuffers;
    vk::raii::CommandPool commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;
    std::vector<vk::raii::Semaphore> imageAvailableSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishSemaphores;
    std::vector<vk::raii::Fence> inFlightFences;

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
            .setPApplicationInfo(&appInfo)
            .setPEnabledExtensionNames(extensions)
            .setPEnabledLayerNames(layers);

        // Create the Vulkan instance.
        try {
            instance = vk::raii::Instance(context, instInfo);
        } catch(const std::exception& e) {
            std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
        }
    }

    void createSurface() {
        // Create a Vulkan surface for rendering
        VkSurfaceKHR c_surface;
        if (!SDL_Vulkan_CreateSurface(window, static_cast<VkInstance>(*instance), &c_surface)) {
            std::cout << "Could not create a Vulkan surface." << std::endl;
            return;
        }
        surface = vk::raii::SurfaceKHR(instance, c_surface);
    }

    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {
            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };

    QueueFamilyIndices findQueueFamilies(const vk::raii::PhysicalDevice& device) {
        QueueFamilyIndices indices;

        int i = 0;
        for (const auto& queue : device.getQueueFamilyProperties()) {
            if (queue.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            if (device.getSurfaceSupportKHR(i, *surface)) {
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

    SwapChainSupportDetails querySwapChainSupport(const vk::raii::PhysicalDevice& device) {
        SwapChainSupportDetails details;

        details.capabilities = device.getSurfaceCapabilitiesKHR(*surface);
        details.formats = device.getSurfaceFormatsKHR(*surface);
        details.presentModes = device.getSurfacePresentModesKHR(*surface);

        return details;
    }

    bool checkDeviceExtensionSupport(const vk::raii::PhysicalDevice& device) {
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : device.enumerateDeviceExtensionProperties()) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    bool isDeviceSuitable(const vk::raii::PhysicalDevice& device) {
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
        for (auto& device : vk::raii::PhysicalDevices(instance)) {
            if (isDeviceSuitable(device)) {
                physicalDevice = std::move(device);
            }
        }
    }

    void createLogicalDevice() {
        float queuePriority = 1.0f;
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<unsigned int> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        for (unsigned int queueFamily : uniqueQueueFamilies) {
            queueCreateInfos.emplace_back(vk::DeviceQueueCreateFlags(), queueFamily, 1, &queuePriority);
        }

        vk::DeviceCreateInfo createInfo = vk::DeviceCreateInfo()
            .setQueueCreateInfos(queueCreateInfos)
            .setPEnabledLayerNames(layers)
            .setPEnabledExtensionNames(deviceExtensions);

        device = vk::raii::Device(physicalDevice, createInfo);
        graphicsQueue = vk::raii::Queue(device, indices.graphicsFamily.value(), 0);
        presentQueue = vk::raii::Queue(device, indices.presentFamily.value(), 0);
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
            .setSurface(*surface)
            .setMinImageCount(imageCount)
            .setImageFormat(surfaceFormat.format)
            .setImageColorSpace(surfaceFormat.colorSpace)
            .setImageExtent(extent)
            .setImageArrayLayers(1)
            .setImageUsage(vk::ImageUsageFlagBits::eColorAttachment)
            .setPreTransform(swapChainSupport.capabilities.currentTransform)
            .setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque)
            .setPresentMode(presentMode)
            .setClipped(true)
            .setOldSwapchain(nullptr);

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<uint32_t> queueFamilyIndices = {indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo
                .setImageSharingMode(vk::SharingMode::eConcurrent)
                .setQueueFamilyIndices(queueFamilyIndices);
        } else {
            createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
        }

        swapChain = vk::raii::SwapchainKHR(device, createInfo);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews() {
        vk::ImageViewCreateInfo createInfo = vk::ImageViewCreateInfo()
            .setViewType(vk::ImageViewType::e2D)
            .setFormat(swapChainImageFormat)
            .setSubresourceRange({ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 });

        for (const auto& swapChainImage : swapChain.getImages()) {
            createInfo.setImage(swapChainImage);
            swapChainImageViews.emplace_back(device, createInfo);
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
            .setColorAttachments(colorAttachmentRef);

        vk::SubpassDependency dependency = vk::SubpassDependency()
            .setSrcSubpass(VK_SUBPASS_EXTERNAL)
            .setDstSubpass(0)
            .setSrcStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setSrcAccessMask(vk::AccessFlagBits::eNoneKHR)
            .setDstStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput)
            .setDstAccessMask(vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo = vk::RenderPassCreateInfo()
            .setAttachments(colorAttachment)
            .setSubpasses(subpass)
            .setDependencies(dependency);

        renderPass = vk::raii::RenderPass(device, renderPassInfo);
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

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) {
        vk::ShaderModuleCreateInfo createInfo = vk::ShaderModuleCreateInfo()
            .setCodeSize(code.size())
            .setPCode(reinterpret_cast<const uint32_t*>(code.data()));

        return vk::raii::ShaderModule(device, createInfo);
    }

    void createGraphicsPipeline() {
        auto vertShaderCode = readFile("vert.spv");
        auto fragShaderCode = readFile("frag.spv");

        vk::raii::ShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        vk::raii::ShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eVertex)
            .setModule(*vertShaderModule)
            .setPName("main");

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo = vk::PipelineShaderStageCreateInfo()
            .setStage(vk::ShaderStageFlagBits::eFragment)
            .setModule(*fragShaderModule)
            .setPName("main");

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = { vertShaderStageInfo, fragShaderStageInfo };

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
            .setViewports(viewport)
            .setScissors(scissor);

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
            .setAttachments(colorBlendAttachment);

        std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eLineWidth };
        vk::PipelineDynamicStateCreateInfo dynamicState = vk::PipelineDynamicStateCreateInfo()
            .setDynamicStates(dynamicStates);

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = vk::PipelineLayoutCreateInfo();
        pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo = vk::GraphicsPipelineCreateInfo()
            .setStages(shaderStages)
            .setPVertexInputState(&vertexInputInfo)
            .setPInputAssemblyState(&inputAssembly)
            .setPViewportState(&viewportState)
            .setPRasterizationState(&rasterizer)
            .setPMultisampleState(&multisampling)
            .setPColorBlendState(&colorBlending)
            .setLayout(*pipelineLayout)
            .setRenderPass(*renderPass)
            .setSubpass(0);
        
        graphicsPipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        for (const auto& swapChainImageView : swapChainImageViews) {
            vk::FramebufferCreateInfo framebufferInfo = vk::FramebufferCreateInfo()
                .setRenderPass(*renderPass)
                .setAttachments(*swapChainImageView)
                .setWidth(swapChainExtent.width)
                .setHeight(swapChainExtent.height)
                .setLayers(1);

            swapChainFrambuffers.emplace_back(device, framebufferInfo);
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
            .setCommandPool(*commandPool)
            .setLevel(vk::CommandBufferLevel::ePrimary)
            .setCommandBufferCount(static_cast<uint32_t>(swapChainFrambuffers.size()));

        commandBuffers = vk::raii::CommandBuffers(device, allocInfo);

        vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo()
            .setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);
        
        vk::ClearValue clearColor = vk::ClearValue()
            .setColor(std::array{ 0.0f, 0.0f, 0.0f, 1.0f });

        for (size_t i = 0; i < commandBuffers.size(); ++i) {
            commandBuffers[i].begin(beginInfo);

            vk::RenderPassBeginInfo renderPassInfo = vk::RenderPassBeginInfo()
                .setRenderPass(*renderPass)
                .setFramebuffer(*swapChainFrambuffers[i])
                .setRenderArea({ {0, 0}, swapChainExtent })
                .setClearValues(clearColor);

            commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
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
            imageAvailableSemaphores.emplace_back(device, semaphoreInfo);
            renderFinishSemaphores.emplace_back(device, semaphoreInfo);
            inFlightFences.emplace_back(device, fenceInfo);
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
        vk::Result result = device.waitForFences(*inFlightFences[currentFrame], true, std::numeric_limits<uint64_t>::max());
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(vk::to_string(result));
        }

        device.resetFences(*inFlightFences[currentFrame]);

        auto [_, imageIndex] = swapChain.acquireNextImage(std::numeric_limits<uint64_t>::max(), *imageAvailableSemaphores[currentFrame]);
        
        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submitInfo = vk::SubmitInfo()
            .setWaitSemaphores(*imageAvailableSemaphores[currentFrame])
            .setWaitDstStageMask(waitStage)
            .setCommandBuffers(*commandBuffers[imageIndex])
            .setSignalSemaphores(*renderFinishSemaphores[currentFrame]);

        graphicsQueue.submit(submitInfo, *inFlightFences[currentFrame]);

        vk::PresentInfoKHR presentInfo = vk::PresentInfoKHR()
            .setWaitSemaphores(*renderFinishSemaphores[currentFrame])
            .setSwapchains(*swapChain)
            .setImageIndices(imageIndex);

        result = presentQueue.presentKHR(presentInfo);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error(vk::to_string(result));
        }

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
        SDL_DestroyWindow(window);
        SDL_Quit();
    }

#if defined(_DEBUG)
    void setupDebugCallback() {
        vk::DebugUtilsMessengerCreateInfoEXT debugInfo = vk::DebugUtilsMessengerCreateInfoEXT()
            .setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo /*| vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose*/)
            .setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation)
            .setPfnUserCallback(debugCallback);
        debugMessager = vk::raii::DebugUtilsMessengerEXT(instance, debugInfo);
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

    vk::raii::DebugUtilsMessengerEXT debugMessager = nullptr;
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
