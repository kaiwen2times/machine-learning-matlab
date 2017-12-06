load 'imagenet-googlenet-dag.mat' %if it's in the path

for i=1:64
    subplot(8,8,i)
    imshow(params(1).value(:,:,:,i));
end;
