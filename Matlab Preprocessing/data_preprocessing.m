clear all 
clc
close all

%% First we need to understand the contents of the dataset %%
addpath('D:\Duke_DME_dataset\2015_BOE_Chiu');
load('Subject_10.mat');
%%%%%%%%%%%%%
%% I'd like to first get all the images with the manual labels%%
%%
% Implementing the labeled-extraction
[labeled_manual1,labeled_index1]=labeled_extraction(manualLayers1);
% Extraction all the files
%% Manual labeled set 2
[labeled_manual2,labeled_index2]=labeled_extraction(manualLayers2);
temp_images=images(:,:,labeled_index2);

%%
file_extraction("Subject_",10);
%%
load('labeled_layers1.mat');
load('labeled_images.mat');
%%
%ploting the ground truth and layers
test_frame=output_images(:,:,1);
test_layers=output_manual_layer1(:,:,1);
imshow(test_frame,[]);
plot_image_layers(test_frame,test_layers);
%%
original=test_layers(5,:);
[~,n]=size(original);
x=1:n;
[F,TF] = fillmissing(original,'linear','SamplePoints',x);
plot(x,original,'.',x(TF),F(TF),'o');
%%
layers1=layers_fill_in(output_manual_layer1);
%layers2=layers_fill_in(output_manual_layer2);
%%
%Transfer layers to regions
% layers to segmentation masks
layers1=round(layers1);
%layers2=round(layers2);
%We need to cut off those areas where it is not atnnotated
masks1=layers_to_masks(output_images(:,117:632,:),layers1(:,117:632,:));
%masks2=layers_to_masks(output_images(:,117:632,:),layers2(:,117:632,:));
%%
weighted_samples=weighted_sampling(masks1);
%%
save('resized_image.mat','resized_images');
save('resized_mask1.mat','masks1');
save('weighted_mask.mat','weighted_samples');
%save('mask2.mat','masks2');
%%
%export as CSV file to load later in Python
images=output_images(:,117:632,:);
csvwrite('masks1.csv',masks1);
%csvwrite('masks2.csv',masks2);
csvwrite('images.csv',images);
            
            
