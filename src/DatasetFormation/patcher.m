close all
clear
clc
%rng(1234)
rng(4567)
%%OPTIONS
writeoutput = true;

patchsize = 128;
num_patches = 4;
lambda = [420:10:700];
augment.scales = [128,512,1024];
augment.shuffles = 3;

%% HDF setup
hdfname = 'I:\AugmentDataset.h5';
if writeoutput
    %h5create(hdfname,'/blurred',[patchsize,patchsize,length(lambda),Inf],'Chunksize',[128,128,length(lambda),128],'Datatype','single','Deflate',0)
    %h5create(hdfname,'/mono',[patchsize,patchsize,Inf],'Chunksize',[128,128,128],'Datatype','single','Deflate',0)
    %h5create(hdfname,'/groundtruth',[patchsize,patchsize,length(lambda),Inf],'Chunksize',[128,128,length(lambda),128],'Datatype','single','Deflate',0)
    %h5create(hdfname,'/coordinate',[2, Inf],'Chunksize',[2,1024],'Datatype','uint16','Deflate',0)
    %h5create(hdfname,'/info',[1,Inf],'Chunksize',[1,1024],'Datatype','string','Deflate',0)
end

%% LOAD PSF


psf_file = 'PSFs\PSF_v1.mat';

load(psf_file);
PSF = single(PSF);
gpuPSF = gpuArray(PSF);

%% CONFIGURE DATA LOADING

foldernames = {'complete_ms_data',...
               'CZ_hsdb',...
               'CZ_hsdbi',...
               'ICVL',...
               'KAIST',...
               'ISET_faces',...
               'ISET_fruit',...
               'ISET_landscape'};
%foldernames = {'complete_ms_data'};

idx.s400 = [3:31];
idx.s420 = [1:29];

%% PROCESSING

%Begin data loading
i = 1;
totalcount = 12609;

for foldername = foldernames
    %iterate over every folder
    
    foldername = foldername{1};
    files = {dir([foldername,'\*.mat']).name};
    
    for filename = files
        tic
        filename = filename{1};
        sampler = false;
        switch i
            case 1
                %break
                load([foldername,'\',filename],'scene');
                hyper = scene;
                clear scene
                keepidx = idx.s400;
                %sampler = true;
            case 2
                %break
                load([foldername,'\',filename],'ref');
                hyper = ref;
                clear ref;
                keepidx = idx.s420;
            case 3
                %break
                load([foldername,'\',filename],'ref');
                hyper = ref;
                clear ref;
                keepidx = idx.s420;
            case 4
                %break
                load([foldername,'\',filename],'rad');
                hyper = rad;
                clear rad;
                keepidx = idx.s400;
            case 5
                %break
                load([foldername,'\',filename],'scene');
                hyper = scene;
                clear scene;
                keepidx = idx.s420;
            case {6,7,8}
                if ~contains(filename,'_Params')
                    load([foldername,'\',filename],'mcCOEF','basis','imgMean')
                    binnedbasis = reshape(interp1(basis.wave,basis.basis,420:1:709,'linear'),10,29,[]);
                    basis.new = reshape(mean(binnedbasis,1),29,[]);
                    binnedmean = mean(reshape(interp1(basis.wave,imgMean,420:1:709,'linear'),10,[]));
                    originalsize = size(mcCOEF);
                    hyper = reshape(reshape(mcCOEF,[],originalsize(3))*basis.new',originalsize(1),originalsize(2),[]) + reshape(binnedmean,1,1,29);
                    hyper = single(hyper);
                    keepidx = idx.s420;

                    clear mcCOEF binnedbasis basis originalsize imgMean
                else
                    continue
                end
                
            otherwise
                erxror('ISET IS JANK');
        end

        fprintf(['LOADED=',foldername,' : ', filename, '\n']);
        hyper = hyper(:,:,keepidx);
        
        hyper = imresize(hyper,256/min(size(hyper,[1,2])));
        hyper = single(hyper)./single(max(hyper,[],'all'));

        hyper = cat(4,hyper,fliplr(hyper),flipud(hyper),fliplr(flipud(hyper)));

        %Initialize GPU vars
        

        gpuHyper = gpuArray(hyper);
        gpumono = gpuArray(zeros(size(hyper,[1,2,4])));

        
        %Main PSF Convolution logic
        
         
        if sampler %create some tiffs and blurred hyper cubes for reference
            blurhyper = zeros(size(hyper));
        end
        blurhyper = zeros(size(hyper));
        for flipz = 1:4
            for wave = 1:length(lambda) %main convolution
                convslice = conv2(gpuHyper(:,:,wave,flipz),gpuPSF(:,:,wave),'same');
                blurhyper(:,:,wave,flipz) = single(gather(convslice));
                if sampler
                    blurhyper(:,:,wave) = single(gather(convslice));
                end
                gpumono(:,:,flipz) = gpumono(:,:,flipz) + convslice;
            end
        end
        mono = single(gather(gpumono));
        mono = mono./max(mono,[],'all');
        blurhyper = blurhyper./max(blurhyper,[],'all');
        if sampler
            blurhyper = blurhyper./max(blurhyper,[],'all');
            save(['examplesamples\',filename(1:end-4),'_blur','.mat'],'blurhyper');
            for j = 1:length(lambda)
                if j == 1
                    % First slice:
                    imwrite(uint16(65535.*blurhyper(:,:,j)),['examplesamples\',filename(1:end-4),'_blur','.tiff'])
                else
                    % Subsequent slices:
                    imwrite(uint16(65535.*blurhyper(:,:,j)),['examplesamples\',filename(1:end-4),'_blur','.tiff'],'WriteMode','append');
                end     
            end

            for j = 1:length(lambda)
                if j == 1
                    % First slice:
                    imwrite(uint16(65535.*hyper(:,:,j)),['examplesamples\',filename(1:end-4),'_GT','.tiff'])
                else
                    % Subsequent slices:
                    imwrite(uint16(65535.*hyper(:,:,j)),['examplesamples\',filename(1:end-4),'_GT','.tiff'],'WriteMode','append');
                end     
            end

            imwrite(uint16(65535.*mono),['examplesamples\',filename(1:end-4),'_mono','.tiff'])
            clear blurhyper
        end

        %Randomly sample 128x128 patches w/ground truth pairs

        for q = 1:num_patches
            for flipselect = 1:4
                patchx = randi([1,size(mono,1)-patchsize]);
                patchy = randi([1,size(mono,2)-patchsize]);
                patchmono = mono(patchx:patchx+patchsize-1,patchy:patchy+patchsize-1,flipselect);
                groundtruth = hyper(patchx:patchx+patchsize-1,patchy:patchy+patchsize-1,:,flipselect);
                blurpatch = single(blurhyper(patchx:patchx+patchsize-1,patchy:patchy+patchsize-1,:,flipselect));
    
                %MAIN DATA WRITING LOGIC
                if writeoutput
                    h5write(hdfname,'/blurred',blurpatch,[1,1,1,totalcount],[patchsize,patchsize,length(lambda),1])
                    h5write(hdfname,'/mono',patchmono,[1, 1, totalcount],[patchsize,patchsize,1])
                    h5write(hdfname,'/groundtruth',groundtruth,[1,1,1,totalcount],[patchsize,patchsize,length(lambda),1])
                    h5write(hdfname,'/coordinate',[uint16(patchx);uint16(patchy)],[1,totalcount],[2,1])
                    h5write(hdfname,'/info',string(['Source: ',foldername,'\',filename,', Info: ','Resized to 512']),[1,totalcount],[1,1])
                end
        
                totalcount = totalcount+1;
            end
        end
        toc
    end
    i = i+1;
end

