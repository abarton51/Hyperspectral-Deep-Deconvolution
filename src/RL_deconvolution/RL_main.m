close all
clear
clc

rng(3)

%% Params
PSFdir = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\PSFs';
datadir = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General';

PSFfile = [PSFdir,'\PSF_v1.mat'];
datafile = [datadir,'\Dataset.h5'];

rawimdim = h5info(datafile,'/mono').Dataspace.MaxSize(1:2);
%this doesn't do anything yet btw; it is in the scenario in which the image
%itself is not a power of two and so must also be embedded... But at the
%moment nothing is being done with it yet.
padimdim = [2^nextpow2(rawimdim(1)),2^nextpow2(rawimdim(2))];
totalsamples = h5info(datafile,'/mono').Dataspace.Size(3);

RLiters = 15;
drawfigs = 1;

redmap = zeros(256,3);
redmap(:,1) = linspace(0,1,256);
greenmap = zeros(256,3);
greenmap(:,2) = linspace(0,1,256);
bluemap = zeros(256,3);
bluemap(:,3) = linspace(0,1,256);

%% Load PSF, embed in padded space, convert to OTF
load([PSFdir,'\PSF_v1.mat'])
%PSF = PSF.*(max(sum(PSF,[1,2]))./sum(PSF,[1,2]));
%PSF = PSF./(max(PSF,[],[1,2]));
OTF = zeros([padimdim(1),padimdim(2),size(PSF,3)]);
IOTF = OTF;

starti = padimdim(1)/2 - floor(size(PSF,1)/2) + 1;
startj = padimdim(2)/2 - floor(size(PSF,1)/2) + 1;
OTF(starti:starti+size(PSF,1)-1,startj:startj+size(PSF,2)-1,:) = PSF;
IOTF(starti:starti+size(PSF,1)-1,startj:startj+size(PSF,2)-1,:) = imrotate(PSF,180);

OTF = fft2(fftshift(OTF));
OTF = gpuArray(single(OTF));
IOTF = fft2(fftshift(IOTF));
IOTF = gpuArray(single(IOTF));

%% RL deconvolution

mono = gpuArray(h5read(datafile,'/mono',[1,1,1],[rawimdim,6304]));
%mono = (mono-min(mono,[],[1,2]))./(max(mono,[],[1,2])-min(mono,[],[1,2]));

j = 1;
selection = randperm(6304,100);
%2499
%2743
%85
%21
for i = [21,85,2499,2743]
    GT = h5read(datafile,'/groundtruth',[1,1,1,i],[128,128,29,1]);
    GTcrop = GT(17:128-16,17:128-16,:);
    %GTcrop = (GTcrop-min(GTcrop,[],'all'))./(max(GTcrop,[],'all')-min(GTcrop,[],'all'));
    tic
    blurred = mono(:,:,i);
    recon = RLcore(OTF,IOTF,blurred,[padimdim,size(PSF,3)],RLiters);
    reconcrop = gather(recon(17:128-16,17:128-16,:));
    blurredcompare = gather(blurred(17:128-16,17:128-16));
    reconcrop = (reconcrop-min(reconcrop,[],'all'))./(max(reconcrop,[],'all')-min(reconcrop,[],'all'));
    %yes the indices are hardcoded for now...
    %btw we crop only the central due to known ringing effects around the
    %border due to the PSF
    toc

    if drawfigs
        B = GTcrop(:,:,1);
        G = GTcrop(:,:,15);
        R = GTcrop(:,:,29);
        Br = reconcrop(:,:,1);
        Gr = reconcrop(:,:,15);
        Rr = reconcrop(:,:,29); 

        figure('Position',[0,0,2000,400])
        subplot(2,8,1)
        imagesc(mean(GTcrop,3));
        axis square
        axis off

        ax = subplot(2,8,2);
        imagesc(R)
        colormap(ax,redmap)
        axis square
        axis off

        ax = subplot(2,8,3);
        imagesc(G)
        colormap(ax,greenmap)
        axis square
        axis off

        ax = subplot(2,8,4);
        imagesc(B)
        colormap(ax,bluemap)
        axis square
        axis off
        
        subplot(2,8,5)
        imagesc(cat(3,R,G,B))
        axis square
        axis off

        subplot(2,8,6)
        tmp = GTcrop(24,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        subplot(2,8,7)
        tmp = GTcrop(48,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        subplot(2,8,8)
        tmp = GTcrop(72,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        subplot(2,8,9)
        imagesc(blurredcompare);
        axis square
        axis off

        ax = subplot(2,8,10);
        imagesc(Rr)
        colormap(ax,redmap)
        axis square
        axis off

        ax = subplot(2,8,11);
        imagesc(Gr)
        colormap(ax,greenmap)
        axis square
        axis off

        ax = subplot(2,8,12);
        imagesc(Br)
        colormap(ax,bluemap)
        axis square
        axis off

        subplot(2,8,13)
        imagesc(cat(3,Rr,Gr,Br))
        axis square
        axis off

        subplot(2,8,14)
        tmp = reconcrop(24,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        subplot(2,8,15)
        tmp = reconcrop(48,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        subplot(2,8,16)
        tmp = reconcrop(72,:,:);
        tmp = reshape(tmp,[96,29]);
        surf([420:10:700],[1:96],tmp,'EdgeAlpha',0);
        xlabel('\lambda')
        ylabel('px')
        zlabel('Intensity')

        drawnow; 

    end

    loss(j) = mean((GTcrop - reconcrop).^2,'all');
    j = j+1;

end

