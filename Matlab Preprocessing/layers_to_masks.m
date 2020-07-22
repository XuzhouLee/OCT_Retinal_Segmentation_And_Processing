function [output_masks] = layers_to_masks(input_images,input_layers)
[m,n,~]=size(input_layers);
[x,y,l]=size(input_images);
output_masks=zeros(x,y,l);
for k=1:l
    test_layers=input_layers(:,:,k);
    i=1;
    for j=1:n
        output_masks(1:test_layers(i,j),j,k)=0*ones(test_layers(i,j),1);
    end
    i=m;
    for j=1:n
    	output_masks(test_layers(i,j):x,j,k)=0*ones(x-test_layers(i,j)+1,1);
    end

    for i=1:m-1
        for j=1:n
            if (test_layers(i,j)>test_layers(i+1,j))
                continue
            end
            output_masks(test_layers(i,j):test_layers(i+1,j),j,k)=i*ones(-test_layers(i,j)+test_layers(i+1,j)+1,1);
        end
    end
end
end

