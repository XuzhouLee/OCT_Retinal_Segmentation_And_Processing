function [] = plot_image_layers(image,layers)
imshow(image,[]);
hold on;
[m,n]=size(layers);
for j=1:m
    plot(1:n,layers(j,:));
end
hold off;
end

