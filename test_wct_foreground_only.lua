require 'torch'
--require 'unsup'
require 'nn'
require 'image'
require 'paths'
require 'lib/utils'
require 'nngraph'
require 'cudnn'
require 'cunn'

local cmd = torch.CmdLine()


--style applied to Foreground (mask==1) only
cmd:option('-style', 'input/style/v2.jpg','two style images, separate by comma')
cmd:option('-mask', 'input/mask/random_mask1.jpg', 'A binary mask to apply spatial control')

cmd:option('-content', 'input/content/004.jpg', 'content image')

cmd:option('-alpha', 0.6)

cmd:option('-vgg1', 'models/vgg_normalised_conv1_1.t7', 'Path to the VGG network')
cmd:option('-vgg2', 'models/vgg_normalised_conv2_1.t7', 'Path to the VGG network')
cmd:option('-vgg3', 'models/vgg_normalised_conv3_1.t7', 'Path to the VGG network')
cmd:option('-vgg4', 'models/vgg_normalised_conv4_1.t7', 'Path to the VGG network')
cmd:option('-vgg5', 'models/vgg_normalised_conv5_1.t7', 'Path to the VGG network')

cmd:option('-decoder5', 'models/feature_invertor_conv5_1.t7', 'Path to the decoder')
cmd:option('-decoder4', 'models/feature_invertor_conv4_1.t7', 'Path to the decoder')
cmd:option('-decoder3', 'models/feature_invertor_conv3_1.t7', 'Path to the decoder')
cmd:option('-decoder2', 'models/feature_invertor_conv2_1.t7', 'Path to the decoder')
cmd:option('-decoder1', 'models/feature_invertor_conv1_1.t7', 'Path to the decoder')

-- Additional options
cmd:option('-contentSize', 512,
           'New (minimum) size for the content image, keeping the original size if set to 0')
cmd:option('-styleSize', 512,
           'New (minimum) size for the style image, keeping the original size if set to 0')
cmd:option('-saveExt', 'jpg', 'The extension name of the output image')
cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
cmd:option('-outputDir', 'output', 'Directory to save the output image(s)')
cmd:option('-saveOriginal', false, 
            'If true, also save the original content and style images in the output directory')

opt = cmd:parse(arg)

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
    cutorch.setDevice(opt.gpu+1)
    --vgg1 = cudnn.convert(vgg1, cudnn):cuda()
    print('GPU')
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


function maximum(a,b)
    if a<b then
        return b
    else
        return a
    end
end


function wct2(contentFeature, styleFeature)

   -- content feature whitening
    local sg = contentFeature:size()
    local c_mean = torch.mean(contentFeature, 2)
    contentFeature = contentFeature - c_mean:expandAs(contentFeature)
    local contentCov = torch.mm(contentFeature, contentFeature:t()):div(sg[2]-1)  --512*512
    local c_e, c_v = torch.symeig(contentCov, 'V')  

    local k_c = 0
    for i=1, sg[1] do
       if c_e[i] > 0.00001 then
            k_c = i
            break
       end
    end

    -- style feature whitening
    local sz = styleFeature:size()
    local styleFeature1 = styleFeature:view(sz[1], sz[2]*sz[3])
    local s_mean = torch.mean(styleFeature1, 2)

    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1)  --512*512
    local s_e, s_v = torch.symeig(styleCov, 'V')

    local k_s = 0
    for i=1, sz[1] do
       if s_e[i] > 0.00001 then
            k_s = i
            break
       end
    end

    local k = maximum(k_c, k_s)
   
    local c_d = c_e[{{k,sg[1]}}]:sqrt():pow(-1)
    local n_contentFeature = c_v[{{},{k,sg[1]}}]*torch.diag(c_d)*(c_v[{{},{k,sg[1]}}]:t())*contentFeature   
    local s_d1 = s_e[{{k,sz[1]}}]:sqrt()
    
    local targetFeature = s_v[{{},{k,sz[1]}}]*(torch.diag(s_d1))*(s_v[{{},{k,sz[1]}}]:t())*n_contentFeature

    targetFeature = targetFeature + s_mean:expandAs(targetFeature)   
    return targetFeature
end


function feature_wct(cF, sF1)
   
   local contentFeature = cF
   local styleFeature1 = sF1

   contentFeature = contentFeature:double()

   styleFeature1 = styleFeature1:double()
   local sz = styleFeature1:size()


    local styleFeatureFG = styleFeature1
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
 
    targetFeature = contentFeatureView:clone():zero() -- C * (H*W)
    targetFeature:indexCopy(2, fgmask, targetFeatureFG)
    targetFeature:indexCopy(2, bgmask, contentFeatureBG)
    targetFeature = targetFeature:viewAs(contentFeature)
   
   
    if opt.gpu >= 0 then
       return targetFeature:cuda()
    else
       return targetFeature:float()
    end
end


local function styleTransfer(content, style)

    local s1 = style[1]

    if opt.gpu >= 0 then
        content = content:cuda()        
        s1 = s1:cuda()
    else
        content = content:float()
        s1 = s1:float()
    end

    local cF5 = vgg5:forward(content):clone()
    local sF51 = vgg5:forward(s1):clone()
    vgg5 = nil
    local csF5 = feature_wct(cF5, sF51)

    csF5 = opt.alpha * csF5 + (1-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil
    ------

    local cF4 = vgg4:forward(Im5):clone()
    local sF41 = vgg4:forward(s1):clone()
    vgg4 = nil

    local csF4 = feature_wct(cF4, sF41)

    csF4 = opt.alpha * csF4 + (1-opt.alpha) * cF4

    local Im4 = decoder4:forward(csF4)
    decoder4 = nil
    -------

    local cF3 = vgg3:forward(Im4):clone()
    local sF31 = vgg3:forward(s1):clone()
    vgg3 = nil
  
    local csF3 = feature_wct(cF3, sF31)
    csF3 = opt.alpha * csF3 + (1-opt.alpha) * cF3
    local Im3 = decoder3:forward(csF3)
    decoder3 = nil
    ----

    local cF2 = vgg2:forward(Im3):clone()
    local sF21 = vgg2:forward(s1):clone()
    vgg2 = nil

    local csF2 = feature_wct(cF2, sF21)
    csF2 = opt.alpha * csF2 + (1-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil
    ----------------

    local cF1 = vgg1:forward(Im2):clone()
    local sF11 = vgg1:forward(s1):clone()
    vgg1 = nil

    local csF1 = feature_wct(cF1, sF11)
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

for i=1,numContent do
    local contentPath = contentPaths[i]
    local contentExt = paths.extname(contentPath)
    local contentImg = image.load(contentPath, 3, 'float')
    local contentName = paths.basename(contentPath, contentExt)
    local contentImg = sizePreprocess(contentImg, opt.contentSize)

    styleImg = {}
    styleName = ''
    for j=1,numStyle do -- generate a transferred image for each (content, style) pair
        local stylePath = stylePaths[j]
        styleExt = paths.extname(stylePath)
        style = image.load(stylePath, 3, 'float')
        style = sizePreprocess(style, opt.styleSize)
        table.insert(styleImg, style)
        styleName = paths.basename(stylePath, styleExt)
    end
    
    local timer = torch.Timer() 
        
    local output = styleTransfer(contentImg, styleImg)

    local C, H, W = output:size(1), output:size(2), output:size(3)
    contentImg = image.scale(contentImg, W, H):cuda()
    local maskResized = image.scale(mask, W, H, 'simple')
    maskResized = torch.gt(maskResized, 0.5):cuda()
    local targetOutput = output

    for cha = 1, C do
       targetOutput[{{cha},{},{}}][1] = torch.cmul(output[{{cha},{},{}}][1], maskResized) + torch.cmul(contentImg[{{cha},{},{}}][1], 1-maskResized)
    end

    print('Time: ' .. timer:time().real .. ' seconds')
    local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_' ..'W_' ..opt.alpha*100 .. '.' .. opt.saveExt)
    print('Output image saved at: ' .. savePath)
    image.save(savePath, output)
    
end