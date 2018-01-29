require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'lib/utils'
require 'nngraph'
require 'cudnn'
require 'cunn'

local cmd = torch.CmdLine()

cmd:option('-style', 'input/style/002937.jpg,input/style/brick.jpg','two style images, separate by comma')
cmd:option('-content', 'input/content/04.jpg', 'content image')

cmd:option('-alpha', 0.6, 'the stylization weight')
cmd:option('-synthesis', 0 , '0-transfer, 1-synthesis')
cmd:option('-beta', 0.5, 'the interpolation weight')

cmd:option('-vgg1', 'models/vgg_normalised_conv1_1.t7', 'Path to the VGG conv1_1')
cmd:option('-vgg2', 'models/vgg_normalised_conv2_1.t7', 'Path to the VGG conv2_1')
cmd:option('-vgg3', 'models/vgg_normalised_conv3_1.t7', 'Path to the VGG conv3_1')
cmd:option('-vgg4', 'models/vgg_normalised_conv4_1.t7', 'Path to the VGG conv4_1')
cmd:option('-vgg5', 'models/vgg_normalised_conv5_1.t7', 'Path to the VGG conv5_1')

cmd:option('-decoder5', 'models/feature_invertor_conv5_1.t7', 'Path to the decoder5')
cmd:option('-decoder4', 'models/feature_invertor_conv4_1.t7', 'Path to the decoder4')
cmd:option('-decoder3', 'models/feature_invertor_conv3_1.t7', 'Path to the decoder3')
cmd:option('-decoder2', 'models/feature_invertor_conv2_1.t7', 'Path to the decoder2')
cmd:option('-decoder1', 'models/feature_invertor_conv1_1.t7', 'Path to the decoder1')

cmd:option('-contentSize', 256, 'New (minimum) size for the content image, keeping the original size if set to 0')
cmd:option('-styleSize', 256, 'New (minimum) size for the style image, keeping the original size if set to 0')
cmd:option('-crop', false, 'If true, center crop both content and style image before resizing')
cmd:option('-saveExt', 'jpg', 'The extension name of the output image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-outputDir', 'output', 'Directory to save the output image(s)')

opt = cmd:parse(arg)

function loadModel()
    vgg1 = torch.load(opt.vgg1)
    vgg2 = torch.load(opt.vgg2)
    vgg3 = torch.load(opt.vgg3)
    vgg4 = torch.load(opt.vgg4)
    vgg5 = torch.load(opt.vgg5)

    decoder5 = torch.load(opt.decoder5)
    decoder4 = torch.load(opt.decoder4)
    decoder3 = torch.load(opt.decoder3)
    decoder2 = torch.load(opt.decoder2)
    decoder1 = torch.load(opt.decoder1)

    if opt.gpu >= 0 then
        print('GPU mode')
        vgg1:cuda()
        vgg2:cuda()
        vgg3:cuda()
        vgg4:cuda()
        vgg5:cuda()
        decoder1:cuda()
        decoder2:cuda()
        decoder3:cuda()
        decoder4:cuda()
        decoder5:cuda()
    else
        print('CPU mode')
        vgg1:float()
        vgg2:float()
        vgg3:float()
        vgg4:float()
        vgg5:float()
        decoder1:float()
        decoder2:float()
        decoder3:float()
        decoder4:float()
        decoder5:float()
    end
end

function feature_decomposition(feature)

    -- feature whitening
    local sg = feature:size()
    local feature1 = feature:view(sg[1], sg[2]*sg[3])
    local c_mean = torch.mean(feature1, 2)
    feature1 = feature1 - c_mean:expandAs(feature1)
    local featureCov = torch.mm(feature1, feature1:t()):div(sg[2]-1)  --512*512
    local c_u, c_e, c_v = torch.svd(featureCov:float(), 'A')  

    local k_c = sg[1]
    for i=1, sg[1] do
       if c_e[i] < 0.00001 then
            k_c = i-1
            break
       end
    end

    return c_e, c_v, k_c, c_mean, feature1
end

function feature_whitening(feature)

    local c_e, c_v, k_c, _, feature1 = feature_decomposition(feature)
    local c_d = c_e[{{1,k_c}}]:sqrt():pow(-1)

    local whiten_feature = nil
    if opt.gpu >= 0 then
        whiten_feature = (c_v[{{},{1,k_c}}]:cuda()) * torch.diag(c_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()) * feature1
    else
        whiten_feature = c_v[{{},{1,k_c}}] * torch.diag(c_d) * (c_v[{{},{1,k_c}}]:t()) * feature1
    end

    return whiten_feature
end


function feature_coloring(whiten_contentFeature, styleFeature)
    
    local s_e, s_v, k_s, s_mean, _ = feature_decomposition(styleFeature)
    local s_d1 = s_e[{{1,k_s}}]:sqrt()
    
    local tFeature = nil
    if opt.gpu >= 0 then
        tFeature = (s_v[{{},{1,k_s}}]:cuda()) * (torch.diag(s_d1:cuda())) * (s_v[{{},{1,k_s}}]:t():cuda()) * whiten_contentFeature
    else 
        tFeature = s_v[{{},{1,k_s}}] * (torch.diag(s_d1)) * (s_v[{{},{1,k_s}}]:t()) * whiten_contentFeature
    end

    tFeature = tFeature + s_mean:expandAs(tFeature)
   
    return tFeature
end

function feature_wct(cF, sF1, sF2)

    local contentFeature = cF
    local styleFeature1 = sF1
    local styleFeature2 = sF2

    local sg = contentFeature:size()

    local whiten_contentFeature = feature_whitening(contentFeature)

    local tFeature1 = feature_coloring(whiten_contentFeature, styleFeature1)
    local tFeature2 = feature_coloring(whiten_contentFeature, styleFeature2)

    local tFeature = opt.beta * tFeature1 + (1 - opt.beta) * tFeature2 
    tFeature = tFeature:resize(sg[1], sg[2], sg[3])

    return tFeature
end


local function styleTransfer(content, style, iteration)

    loadModel()

    local s1 = style[1]
    local s2 = style[2]

    if opt.gpu >= 0 then
        content = content:cuda()
        if opt.synthesis == 1 and iteration == 1 then
           local gg = content:size()
           content = torch.zeros(gg[1],gg[2],gg[3]):uniform():cuda()
           print('input noise for synthesis')
        end
        s1 = s1:cuda()
        s2 = s2:cuda()
    else
        content = content:float()
        if opt.synthesis == 1 and iteration == 1 then
           local gg = content:size()
           content = torch.zeros(gg[1],gg[2],gg[3]):uniform():float()
           print('input noise for synthesis')
        end
        s1 = s1:float()
        s2 = s2:float()
    end

    --WCT on conv5_1
    local cF5 = vgg5:forward(content):clone()
    local sF51 = vgg5:forward(s1):clone()
    local sF52 = vgg5:forward(s2):clone()
    vgg5 = nil
    local csF5 = feature_wct(cF5, sF51, sF52)

    csF5 = opt.alpha * csF5 + (1-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    --WCT on conv4_1
    local cF4 = vgg4:forward(Im5):clone()
    local sF41 = vgg4:forward(s1):clone()
    local sF42 = vgg4:forward(s2):clone()
    vgg4 = nil

    local csF4 = feature_wct(cF4, sF41, sF42)

    csF4 = opt.alpha * csF4 + (1-opt.alpha) * cF4

    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    --WCT on conv3_1
    local cF3 = vgg3:forward(Im4):clone()
    local sF31 = vgg3:forward(s1):clone()
    local sF32 = vgg3:forward(s2):clone()
    vgg3 = nil
  
    local csF3 = feature_wct(cF3, sF31, sF32)
    csF3 = opt.alpha * csF3 + (1-opt.alpha) * cF3
    local Im3 = decoder3:forward(csF3)
    decoder3 = nil

    --WCT on conv2_1
    local cF2 = vgg2:forward(Im3):clone()
    local sF21 = vgg2:forward(s1):clone()
    local sF22 = vgg2:forward(s2):clone()
    vgg2 = nil

    local csF2 = feature_wct(cF2, sF21, sF22)
    csF2 = opt.alpha * csF2 + (1-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    --WCT on conv1_1
    local cF1 = vgg1:forward(Im2):clone()
    local sF11 = vgg1:forward(s1):clone()
    local sF12 = vgg1:forward(s2):clone()
    vgg1 = nil

    local csF1 = feature_wct(cF1, sF11, sF12)
    csF1 = opt.alpha * csF1 + (1-opt.alpha) * cF1
    local Im1 = decoder1:forward(csF1) 
    decoder1 = nil
    
    return Im1
end

print('Creating save folder at ' .. opt.outputDir)
paths.mkdir(opt.outputDir)

local contentPaths = {}
local stylePaths = {}

if opt.content ~= '' then -- use a single content image
    table.insert(contentPaths, opt.content)
else -- use a batch of content images
    assert(opt.contentDir ~= '', "Either opt.contentDir or opt.content should be non-empty!")
    contentPaths = extractImageNamesRecursive(opt.contentDir)
end

if opt.style ~= '' then 
    style_image_list = opt.style:split(',')
    for i=1, #style_image_list do
        table.insert(stylePaths, style_image_list[i])
    end
end

local numContent = #contentPaths
local numStyle = #stylePaths
print("# Content images: " .. numContent)
print("# Style images: " .. numStyle)
print("Make sure that you specify two style images.")

for i=1,numContent do
    local contentPath = contentPaths[i]
    local contentExt = paths.extname(contentPath)
    local contentImg = image.load(contentPath, 3, 'float')
    local contentName = paths.basename(contentPath, contentExt)
    local contentImg = sizePreprocess(contentImg, opt.contentSize)

    styleImg = {}
    styleName = ''
    for j=1,numStyle do 
        local stylePath = stylePaths[j]
        styleExt = paths.extname(stylePath)
        style = image.load(stylePath, 3, 'float')
        style = sizePreprocess(style, opt.styleSize)
        table.insert(styleImg, style)
        styleName = paths.basename(stylePath, styleExt)
    end

    local output = nil
    if opt.synthesis == 0 then
        output = styleTransfer(contentImg, styleImg, 1)
        local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_interpolation_' ..'weight_' ..opt.beta*100 .. '.' .. opt.saveExt)
        print('Output image saved at: ' .. savePath)
        image.save(savePath, output)
    else
        --We empirically find that 3 iterations are enough for a visually-pleasing result
        --By default, for texture synthesis, we set alpha = 1.0 because there are no content features here to blend
        opt.alpha = 1.0
        for iter = 1, 3 do
            output = styleTransfer(contentImg, styleImg, iter)
            local savePath = paths.concat(opt.outputDir, contentName .. '_iter' .. iter .. '_beta_' ..opt.beta*100 .. '.' .. opt.saveExt)
            print('Output image saved at: ' .. savePath)
            image.save(savePath, output)
            contentImg = output
        end
    end
    
end
