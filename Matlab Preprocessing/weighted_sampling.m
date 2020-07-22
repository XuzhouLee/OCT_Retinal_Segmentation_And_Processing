function [output]=weighted_sampling(input_image)
    [m,n,l]=size(input_image);
    output=zeros(m,n,l);
    for i =1:l
        image=input_image(:,:,i);
        for j =1:m
            for k=1:n
                if (image(j,k)==1)
                    w2=11.459;
                elseif (image(j,k)==2)
                    w2=5.63;
                elseif (image(j,k)==3)
                    w2=11.007;
                elseif (image(j,k)==4)
                    w2=14.368;
                elseif (image(j,k)==5)
                    w2=3.336;
                elseif (image(j,k)==6)
                    w2=13.647;
                elseif (image(j,k)==7)
                    w2=16.978;
                else
                    w2=0;
                end
                if (j~=0 && j~=m)
                    if (image(j+1,k)-image(j,k)>0 && w2~=0)
                        w1=15;
                    else
                        w1=0;
                    end
                else
                    w1=0;
                end
                w=1+w1+w2;
                output(j,k,i)=w;
                    
            end
        end
    end         
end
