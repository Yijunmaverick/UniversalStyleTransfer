require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'lib/utils'
require 'nngraph'
require 'cudnn'
require 'cunn'

local cmd = torch.CmdLine()

--1st style applied to Foreground (mask==1), 2nd style applied to Background (mask==0)
cmd:option('-style', 'input/style/088.jpg,input/style/10.jpg','two style images, separate by comma')
cmd:option('-mask', 'input/mask/random_mask3.jpg', 'A binary mask to apply spatial control')

cmd:option('-content', 'input/content/04.jpg', 'content image')

cmd:option('-alpha', 0.6)

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

cmd:option('-contentSize', 768, 'New (minimum) size for the content image, keeping the original size if set to 0')
cmd:option('-styleSize', 768, 'New (minimum) size for the style image, keeping the original size if set to 0')
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


function wct2(contentFeature, styleFeature)

   -- content feature whitening
    local sg = contentFeature:size()
    local c_mean = torch.mean(contentFeature, 2)
    contentFeature = contentFeature - c_mean:expandAs(contentFeature)
    local contentCov = torch.mm(contentFeature, contentFeature:t()):div(sg[2]-1)  --512*512
    local c_u, c_e, c_v = torch.svd(contentCov:float(), 'A')  

    local k_c = sg[1]
    for i=1, sg[1] do
       if c_e[i] < 0.00001 then
            k_c = i-1
            break
       end
    end

    -- style feature whitening
    local sz = styleFeature:size()
    local styleFeature1 = styleFeature:view(sz[1], sz[2]*sz[3])
    local s_mean = torch.mean(styleFeature1, 2)

    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1)  --512*512
    local s_u, s_e, s_v = torch.svd(styleCov:float(), 'A')

    local k_s = sz[1]
    for i=1, sz[1] do
       if s_e[i] < 0.00001 then
            k_s = i-1
            break
       end
    end
    
    
    local c_d = c_e[{{1,k_c}}]:sqrt():pow(-1)
    local s_d1 = s_e[{{1,k_s}}]:sqrt()

    local whiten_contentFeature = nil
    local tFeature = nil
    if opt.gpu >= 0 then
        whiten_contentFeature = (c_v[{{},{1,k_c}}]:cuda()) * torch.diag(c_d:cuda()) * (c_v[{{},{1,k_c}}]:t():cuda()) *contentFeature     
        tFeature = (s_v[{{},{1,k_s}}]:cuda()) * (torch.diag(s_d1:cuda())) * (s_v[{{},{1,k_s}}]:t():cuda()) * whiten_contentFeature
    else
        whiten_contentFeature = c_v[{{},{1,k_c}}] * torch.diag(c_d) * (c_v[{{},{1,k_c}}]:t()) * contentFeature     
        tFeature = s_v[{{},{1,k_s}}] * (torch.diag(s_d1)) * (s_v[{{},{1,k_s}}]:t()) * whiten_contentFeature
    end
    
    tFeature = tFeature + s_mean:expandAs(tFeature)

    return tFeature
end


function feature_wct(contentFeature, styleFeature1, styleFeature2)

    local sz = styleFeature1:size()

    local styleFeatureFG = styleFeature1
    local styleFeatureBG = styleFeature2
    local C, H, W = contentFeature:size(1), contentFeature:size(2), contentFeature:size(3)
    local maskResized = image.scale(mask, W, H, 'simple')

    
    local maskView = maskResized:view(-1)
    maskView = torch.gt(maskView, 0.5)
    local fgmask = torch.LongTensor(torch.find(maskView, 1)) -- foreground indices 
    local bgmask = torch.LongTensor(torch.find(maskView, 0)) -- background indices
        
    local contentFeatureView = contentFeature:view(C, -1)
    local contentFeatureFG = contentFeatureView:index(2, fgmask):view(C, fgmask:nElement()) -- C * #fg
    local contentFeatureBG = contentFeatureView:index(2, bgmask):view(C, bgmask:nElement()) -- C * #bg

    local targetFeatureFG = wct2(contentFeatureFG, styleFeatureFG)
    local targetFeatureBG = wct2(contentFeatureBG, styleFeatureBG)

 
    targetFeature = contentFeatureView:clone():zero() -- C * (H*W)
    targetFeature:indexCopy(2, fgmask ,targetFeatureFG)
    targetFeature:indexCopy(2, bgmask ,targetFeatureBG)
    targetFeature = targetFeature:viewAs(contentFeature)
   
    return targetFeature
end


local function styleTransfer(content, style)

    loadModel()

    local s1 = style[1]
    local s2 = style[2]

    if opt.gpu >= 0 then
        content = content:cuda()        
        s1 = s1:cuda()
        s2 = s2:cuda()
    else
        content = content:float()
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

if opt.mask ~= '' then
    mask = image.load(opt.mask, 1, 'float') -- binary mask
end

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
            
    local output = styleTransfer(contentImg, styleImg)
    local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_' ..'alpha_' ..opt.alpha*100 .. '.' .. opt.saveExt)
    print('Output image saved at: ' .. savePath)
    image.save(savePath, output)
    
end
