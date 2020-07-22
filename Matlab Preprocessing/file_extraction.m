function [] = file_extraction(file_name,file_num)
output_images=[];
output_manual_layer1=[];
output_manual_layer2=[];
for i=1:file_num
    if i<10
        temp_file_name=file_name+"0"+string(i)+".mat";
        load(temp_file_name);
        [labeled_manual1,~]=labeled_extraction(manualLayers1);
        [labeled_manual2,labeled_index2]=labeled_extraction(manualLayers2);
        temp_images=images(:,:,labeled_index2);
        output_manual_layer1=cat(3,output_manual_layer1,labeled_manual1);
        output_manual_layer2=cat(3,output_manual_layer2,labeled_manual2);
        output_images=cat(3,output_images,temp_images);
    else
        temp_file_name=file_name+string(i)+".mat";
        load(temp_file_name);
        [labeled_manual1,~]=labeled_extraction(manualLayers1);
        [labeled_manual2,labeled_index2]=labeled_extraction(manualLayers2);
        temp_images=images(:,:,labeled_index2);
        output_manual_layer1=cat(3,output_manual_layer1,labeled_manual1);
        output_manual_layer2=cat(3,output_manual_layer2,labeled_manual2);
        output_images=cat(3,output_images,temp_images);
    end
   
end
save("labeled_images.mat","output_images");
save("labeled_layers1.mat","output_manual_layer1");
save("labeled_layers2.mat","output_manual_layer2");
end


