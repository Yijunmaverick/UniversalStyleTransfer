require 'nn'
require 'lfs'

-- Prepares an RGB image in [0,1] for VGG
function getPreprocessConv()
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local conv = nn.SpatialConvolution(3,3, 1,1)
    conv.weight:zero()
    conv.weight[{1,3}] = 255
    conv.weight[{2,2}] = 255
    conv.weight[{3,1}] = 255
    conv.bias = -mean_pixel
    conv.gradBias = nil
    conv.gradWeight = nil
    conv.parameters = function() --[[nop]] end
    conv.accGradParameters = function() --[[nop]] end
    return conv
end

function extractImageNamesRecursive(dir)
    local files = {}
    print("Extracting image paths: " .. dir)
  
    local function browseFolder(root, pathTable)
        for entity in lfs.dir(root) do
            if entity~="." and entity~=".." then
                local fullPath=root..'/'..entity
                local mode=lfs.attributes(fullPath,"mode")
                if mode=="file" then
                    local filepath = paths.concat(root, entity)
  
                    if string.find(filepath, 'jpg$')
                    or string.find(filepath, 'png$')
                    or string.find(filepath, 'jpeg$')
                    or string.find(filepath, 'JPEG$')
                    or string.find(filepath, 'ppm$') then
                        table.insert(pathTable, filepath)
                    end
                elseif mode=="directory" then
                    browseFolder(fullPath, pathTable);
                end
            end
        end
    end

    browseFolder(dir, files)
    return files
end


-- image size preprocessing
function sizePreprocess(x, newSize)
    assert(x:dim() == 3)
    if newSize ~= 0 then
        --x = image.scale(x, '^' .. newSize)
        x = image.scale(x, newSize, 'bilinear')
    end
    return x
end


-- copied from torchx: https://github.com/nicholas-leonard/torchx/blob/master/find.lua
function torch.find(tensor, val, dim)
   local i = 1
   local indice = {}
   if dim then
      assert(tensor:dim() == 2, "torch.find dim arg only supports matrices for now")
      assert(dim == 2, "torch.find only supports dim=2 for now")
            
      local colSize, rowSize = tensor:size(1), tensor:size(2)
      local rowIndice = {}
      tensor:apply(function(x)
            if x == val then
               table.insert(rowIndice, i)
            end
            if i == rowSize then
               i = 1
               table.insert(indice, rowIndice)
               rowIndice = {}
            else
               i = i + 1
            end
         end)
   else
      tensor:apply(function(x)
         if x == val then
            table.insert(indice, i)
         end
         i = i + 1
      end)
   end
   return indice
end

------------------------------------------------
