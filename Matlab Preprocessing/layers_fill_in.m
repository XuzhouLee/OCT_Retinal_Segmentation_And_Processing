function [output_sets] = layers_fill_in(input_sets)
[m,n,l]=size(input_sets);
output_sets=zeros(m,n,l);
x=1:n;
for i=1:l
    for j=1:m
        temp_layer=input_sets(j,:,i);
        [F,TF] = fillmissing(temp_layer,'linear','SamplePoints',x);
        output_sets(j,:,i)=F;
    end
end
end

