require 'torch'
require 'nn'
require 'image'
require 'paths'
require 'lib/AdaptiveInstanceNormalization'
require 'lib/utils'
require 'nngraph'
require 'cudnn'
require 'cunn'

local cmd = torch.CmdLine()

cmd:option('-style', 'input/style/09.jpg', 'path to the style image')
cmd:option('-styleDir', '', 'path to the style image folder')
cmd:option('-content', 'input/content/04.jpg', 'path to the content image')
cmd:option('-contentDir', '', 'path to the content image folder')

cmd:option('-alpha', 0.6)
cmd:option('-synthesis', 0 , '0-transfer, 1-synthesis')

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

assert(opt.content ~= '' or opt.contentDir ~= '', 'Either --content or --contentDir should be given.')
assert(opt.style ~= '' or opt.styleDir ~= '', 'Either --style or --styleDir should be given.')
assert(opt.content == '' or opt.contentDir == '', '--content and --contentDir cannot both be given.')
assert(opt.style == '' or opt.styleDir == '', '--style and --styleDir cannot both be given.')


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

    adain5 = nn.AdaptiveInstanceNormalization(vgg5:get(#vgg5-1).nOutputPlane)
    adain4 = nn.AdaptiveInstanceNormalization(vgg4:get(#vgg4-1).nOutputPlane)
    adain3 = nn.AdaptiveInstanceNormalization(vgg3:get(#vgg3-1).nOutputPlane)
    adain2 = nn.AdaptiveInstanceNormalization(vgg2:get(#vgg2-1).nOutputPlane)
    adain1 = nn.AdaptiveInstanceNormalization(vgg1:get(#vgg1-1).nOutputPlane)

    if opt.gpu >= 0 then
        print('GPU mode')
        vgg1:cuda()
        vgg2:cuda()
        vgg3:cuda()
        vgg4:cuda()
        vgg5:cuda()
        adain5:cuda()
	adain4:cuda()
	adain3:cuda()
	adain2:cuda()
	adain1:cuda()
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
        adain5:float()
	adain4:float()
	adain3:float()
	adain2:float()
	adain1:float()
        decoder1:float()
        decoder2:float()
        decoder3:float()
        decoder4:float()
        decoder5:float()
    end
end


local function styleTransfer(content, style, iteration)

    --We delete the model after using it to save momery for performing transfer on large image size
    --So we need to reload it each time for a new round of transfer
    loadModel()

    if opt.gpu >= 0 then
        content = content:cuda()
        if opt.synthesis == 1 and iteration == 1 then
           local gg = content:size()
           content = torch.zeros(gg[1],gg[2],gg[3]):uniform():cuda()
           print('input noise for synthesis')
        end
        style = style:cuda()
    else
        content = content:float()
        if opt.synthesis == 1 and iteration == 1 then
           local gg = content:size()
           content = torch.zeros(gg[1],gg[2],gg[3]):uniform():float()
           print('input noise for synthesis')
        end
        style = style:float()
    end


    --AdaIN on conv5_1
    local cF5 = vgg5:forward(content):clone()
    local sF5 = vgg5:forward(style):clone()
    vgg5 = nil
 
    csF5 = adain5:forward({cF5, sF5}):squeeze()

    csF5 = opt.alpha * csF5 + (1.0-opt.alpha) * cF5
    local Im5 = decoder5:forward(csF5)
    decoder5 = nil

    --AdaIN on conv4_1
    local cF4 = vgg4:forward(Im5):clone()
    local sF4 = vgg4:forward(style):clone()
    vgg4 = nil

    local csF4 = adain4:forward({cF4, sF4}):squeeze()
    csF4 = opt.alpha * csF4 + (1.0-opt.alpha) * cF4
    local Im4 = decoder4:forward(csF4)
    decoder4 = nil

    --AdaIN on conv3_1
    local cF3 = vgg3:forward(Im4):clone()
    local sF3 = vgg3:forward(style):clone()
    vgg3 = nil

    local csF3 = adain3:forward({cF3, sF3}):squeeze()
    csF3 = opt.alpha * csF3 + (1.0-opt.alpha) * cF3 

    local Im3 = decoder3:forward(csF3)
    decoder3 = nil

    --AdaIN on conv2_1
    local cF2 = vgg2:forward(Im3):clone()
    local sF2 = vgg2:forward(style):clone()
    vgg2 = nil

    local csF2 = adain2:forward({cF2, sF2}):squeeze()
    csF2 = opt.alpha * csF2 + (1.0-opt.alpha) * cF2

    local Im2 = decoder2:forward(csF2)
    decoder2 = nil

    --AdaIN on conv1_1
    local cF1 = vgg1:forward(Im2):clone()
    local sF1 = vgg1:forward(style):clone()
    vgg1 = nil

    local csF1 = adain1:forward({cF1, sF1}):squeeze()
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
print("# Content images: " .. numContent)
print("# Style images: " .. numStyle)


for i=1,numContent do
    local contentPath = contentPaths[i]
    local contentExt = paths.extname(contentPath)
    local contentImg = image.load(contentPath, 3, 'float')
    local contentName = paths.basename(contentPath, contentExt)
    local contentImg = sizePreprocess(contentImg, opt.contentSize)

    for j=1,numStyle do

        local stylePath = stylePaths[j]
  
        styleExt = paths.extname(stylePath)
        styleImg = image.load(stylePath, 3, 'float')
        styleImg = sizePreprocess(styleImg, opt.styleSize)
        styleName = paths.basename(stylePath, styleExt)

	local output = nil
        if opt.synthesis == 0 then
            output = styleTransfer(contentImg, styleImg, 1)
            local savePath = paths.concat(opt.outputDir, contentName .. '_stylized_by_' .. styleName .. '_alpha_' ..opt.alpha*100 .. '_adain.' .. opt.saveExt)
            print('Output image saved at: ' .. savePath)
            image.save(savePath, output)

       else
            --We empirically find that 3 iterations are enough for a visually-pleasing result
	    --By default, for texture synthesis, we set alpha = 1.0 because there are no content features here to blend
            opt.alpha = 1.0
            for iter = 1, 3 do
                output = styleTransfer(contentImg, styleImg, iter)
                local savePath = paths.concat(opt.outputDir, styleName .. '_iter' .. iter .. '_alpha_' ..opt.alpha*100 .. '_adain.' .. opt.saveExt)
                print('Output image saved at: ' .. savePath)
                image.save(savePath, output)
                contentImg = output
            end
       end

    end
end
