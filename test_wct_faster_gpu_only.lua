require 'torch'
--require 'unsup'
require 'nn'
require 'image'
require 'paths'
require 'lib/NonparametricPatchAutoencoderFactory'
require 'lib/MaxCoord'
require 'lib/utils'
require 'nngraph'
require 'cudnn'
require 'cunn'

local cmd = torch.CmdLine()

-- Basic options
cmd:option('-style', 'input/style/v2.jpg',
           'File path to the style image, or multiple style images separated by commas if you want to do style interpolation or spatial control')
cmd:option('-styleDir', '', 'Directory path to a batch of style images')
cmd:option('-content', 'input/content/004.jpg', 'File path to the content image')
cmd:option('-contentDir', '', 'Directory path to a batch of content images')

cmd:option('-swap5', 0)
cmd:option('-alpha', 0.6)
cmd:option('-patchSize', 3)
cmd:option('-patchStride', 1)

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

cmd:option('-contentSize', 768,
           'New (minimum) size for the content image, keeping the original size if set to 0')
cmd:option('-styleSize', 512,
           'New (minimum) size for the style image, keeping the original size if set to 0')

cmd:option('-saveExt', 'jpg', 'The extension name of the output image')
--cmd:option('-gpu', 0, 'Zero-indexed ID of the GPU to use; for CPU mode set -gpu = -1')
--'Using CUDA_VISIBLE_DEVICES=XXX instead to ensure all weights/gradients/input are located on the same GPU'
cmd:option('-outputDir', 'output', 'Directory to save the output image(s)')
cmd:option('-saveOriginal', false, 
            'If true, also save the original content and style images in the output directory')

opt = cmd:parse(arg)

assert(opt.content ~= '' or opt.contentDir ~= '', 'Either --content or --contentDir should be given.')
assert(opt.style ~= '' or opt.styleDir ~= '', 'Either --style or --styleDir should be given.')
assert(opt.content == '' or opt.contentDir == '', '--content and --contentDir cannot both be given.')
assert(opt.style == '' or opt.styleDir == '', '--style and --styleDir cannot both be given.')

function maximum(a,b)
    if a<b then
        return b
    else
        return a
    end
end

function feature_swap_whiten(cF, sF)
   
   local contentFeature = cF
   local styleFeature = sF

   -- content feature whitening
    local sg = contentFeature:size()
    local contentFeature1 = contentFeature:view(sg[1], sg[2]*sg[3])
    local c_mean = torch.mean(contentFeature1, 2)
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(sg[2]*sg[3]-1)  --512*512
    local c_e, c_v = torch.symeig(contentCov:float(), 'V')  

    local k_c = 0
    for i=1, sg[1] do
       if c_e[i] > 0.00001 then
            k_c = i
            break
       end
    end
    --print('k_c = ', k_c)

    -- style feature whitening
    local sz = styleFeature:size()
    local styleFeature1 = styleFeature:view(sz[1], sz[2]*sz[3])
    local s_mean = torch.mean(styleFeature1, 2)
    styleFeature1 = styleFeature1 - s_mean:expandAs(styleFeature1)
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1)  --512*512
    local s_e, s_v = torch.symeig(styleCov:float(), 'V')

    local k_s = 0
    for i=1, sz[1] do
       if s_e[i] > 0.00001 then
            k_s = i
            break
       end
    end
    --print('k_s = ', k_s)

    local k = maximum(k_c, k_s)
    --print('k = ', k)


    c_v = c_v:cuda()
    s_v = s_v:cuda()

    local s_d = torch.sqrt(s_e[{{k,sz[1]}}]):pow(-1)
    local diag_s = torch.diag(s_d):cuda()
    local n_styleFeature = s_v[{{},{k,sz[1]}}]*diag_s*(s_v[{{},{k,sz[1]}}]:t())*styleFeature1
    

    local swap_enc, swap_dec = NonparametricPatchAutoencoderFactory.buildAutoencoder(n_styleFeature:resize(sz[1], sz[2], sz[3]), opt.patchSize, opt.patchStride, false, false, true)

    local swap = nn.Sequential()
    swap:add(swap_enc)
    swap:add(nn.MaxCoord())
    swap:add(swap_dec)
    swap:cuda():evaluate()

    local c_d = c_e[{{k,sg[1]}}]:sqrt():pow(-1)
    local diag_c = torch.diag(c_d):cuda()
    local n_contentFeature = c_v[{{},{k,sg[1]}}]*diag_c*(c_v[{{},{k,sg[1]}}]:t())*contentFeature1
       
    local swap_latent = swap:forward(n_contentFeature:resize(sg[1], sg[2], sg[3])):clone()
    local swap_latent1 = swap_latent:view(sg[1], sg[2]*sg[3])

    local s_d1 = torch.sqrt(s_e[{{k,sz[1]}}])
    local diag_s1 = torch.diag(s_d1):cuda()
    local targetFeature = s_v[{{},{k,sz[1]}}]*diag_s1*(s_v[{{},{k,sz[1]}}]:t())*swap_latent1

    targetFeature = targetFeature + s_mean:expandAs(targetFeature)

    local tFeature = targetFeature:resize(sg[1], sg[2], sg[3])
    return tFeature
end


function feature_wct(cF, sF)

   
   local contentFeature = cF
   local styleFeature = sF 

   -- content feature whitening
    local sg = contentFeature:size()
    local contentFeature1 = contentFeature:view(sg[1], sg[2]*sg[3])
    local c_mean = torch.mean(contentFeature1, 2)
    contentFeature1 = contentFeature1 - c_mean:expandAs(contentFeature1)
    local contentCov = torch.mm(contentFeature1, contentFeature1:t()):div(sg[2]*sg[3]-1)    --512*512
    local c_e, c_v = torch.symeig(contentCov:float(), 'V')  

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
    local styleCov = torch.mm(styleFeature1, styleFeature1:t()):div(sz[2]*sz[3]-1) --512*512
    local s_e, s_v = torch.symeig(styleCov:float(), 'V')

    local k_s = 0
    for i=1, sz[1] do
       if s_e[i] > 0.00001 then
            k_s = i
            break
       end
    end

    local k = maximum(k_c, k_s)
    --print('k = ', k)

    c_v = c_v:cuda()
    s_v = s_v:cuda()
    local c_d = c_e[{{k,sg[1]}}]:sqrt():pow(-1)
    local diag_c = torch.diag(c_d):cuda()

    local s_d1 = s_e[{{k,sz[1]}}]:sqrt()
    local diag_s = torch.diag(s_d1):cuda()

 
    local n_contentFeature = c_v[{{},{k,sg[1]}}]*diag_c*(c_v[{{},{k,sg[1]}}]:t())*contentFeature1
    local targetFeature = s_v[{{},{k,sz[1]}}]*diag_s*(s_v[{{},{k,sz[1]}}]:t())*n_contentFeature

    targetFeature = targetFeature + s_mean:expandAs(targetFeature)

    local tFeature = targetFeature:resize(sg[1], sg[2], sg[3])
    return tFeature
end


local function styleTransfer(content, style)

    content = content:cuda()
    style = style:cuda()

    ---Start
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil

    local csF5 = nil
    if opt.swap5 ~= 0 then
        local timer = torch.Timer()
        print('swap on conv5 whiten feature')       
        csF5 = feature_swap_whiten(cF5, sF5)
        print('Swap5 Time: ' .. timer:time().real .. ' seconds')
    else
        local timer = torch.Timer()
        print('wct on conv5 whiten feature')       
        csF5 = feature_wct(cF5, sF5)
        print('WCT5 Time: ' .. timer:time().real .. ' seconds') 
    end

    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil
    ------

    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil
    --local sF4 = vgg4:forward(style):clone():float()

    --local timer = torch.Timer()       
    --print('WCT on conv4 whiten feature')
    local csF4 = feature_wct(cF4, sF4)
    --print('WCT Time: ' .. timer:time().real .. ' seconds')

    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4

    local Im4 = decoder4:forward(csF4)
    decoder4 = nil
    -------

    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil
    --local sF3 = vgg3:forward(style):clone():float()

    --local timer = torch.Timer() 
    --print('WCT on conv3 whiten feature')
    local csF3 = feature_wct(cF3, sF3)
    --print('WCT Time: ' .. timer:time().real .. ' seconds')

    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3
    local Im3 = decoder3:forward(csF3)
    decoder3 = nil
    ----

    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil
    --local sF2 = vgg2:forward(style):clone():float()

    --local timer = torch.Timer()
    --print('WCT on conv2 whiten feature') 
    local csF2 = feature_wct(cF2, sF2)
    --print('WCT Time: ' .. timer:time().real .. ' seconds')

    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2
    local Im2 = decoder2:forward(csF2)
    decoder2 = nil
    ----------------

    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil
    --local sF1 = vgg1:forward(style):clone():float()

    --local timer = torch.Timer() 
    --print('WCT on conv1 whiten feature')
    local csF1 = feature_wct(cF1, sF1)
    --print('WCT Time: ' .. timer:time().real .. ' seconds')

    csF1 = opt.alpha * csF1 + (1.0-opt.alpha) * cF1
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
    if #style_image_list == 1 then
        style_image_list = style_image_list[1]
    end
    table.insert(stylePaths, style_image_list)
else -- use a batch of style images
    assert(opt.styleDir ~= '', "Either opt.styleDir or opt.style should be non-empty!")
    stylePaths = extractImageNamesRecursive(opt.styleDir)
end

local numContent = #contentPaths
local numStyle = #stylePaths
--print("# Content images: " .. numContent)
--print("# Style images: " .. numStyle)


for i=1,numContent do
    local contentPath = contentPaths[i]
    local contentExt = paths.extname(contentPath)
    local contentImg = image.load(contentPath, 3, 'float')
    local contentName = paths.basename(contentPath, contentExt)
    local contentImg = sizePreprocess(contentImg, opt.contentSize)

    for j=1,numStyle do -- generate a transferred image for each (content, style) pair


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

        --cutorch.setDevice(opt.gpu+1)
        --print('GPU')
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

        local stylePath = stylePaths[j]
        
        styleExt = paths.extname(stylePath)
        styleImg = image.load(stylePath, 3, 'float')
        styleImg = sizePreprocess(styleImg, opt.styleSize)
        if opt.preserveColor then
            styleImg = coral(styleImg, contentImg)
        end
        styleName = paths.basename(stylePath, styleExt)
     
        local output = nil
       
        local timer = torch.Timer() 
        output = styleTransfer(contentImg, styleImg)
        print('Time: ' .. timer:time().real .. ' seconds')
        local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_' .. styleName .. '_W_' ..opt.alpha*100 .. '.' .. opt.saveExt)
        print('Output image saved at: ' .. savePath)
        image.save(savePath, output)

        if opt.outputDirOriginal then
            -- also save the original images
            image.save(paths.concat(opt.outputDir, contentName .. '.' .. contentExt), contentImg)
            image.save(paths.concat(opt.outputDir, styleName .. '.' .. styleExt), styleImg)
        end
    end
end
