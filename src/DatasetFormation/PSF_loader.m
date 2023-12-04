close all
clear
clc

PSFdir = 'C:\Users\czheng45\Documents\CoreyOptics\CS7643\PSF1';
outname = 'PSF_v1';

slices = {dir([PSFdir,'\*.txt']).name};

spacingline = 9;
center.r = 129;
center.c = 128;
pxsz = 1.4;
kernelsz = 15;
PSF = zeros(kernelsz*2+1,kernelsz*2+1,29);
i = 1;

iwrite = true;

figure
for filename = slices
    filename = filename{1};
    fid = fopen([PSFdir,'\',filename]);

    PSFlayer = readmatrix([PSFdir,'\',filename]);

    gridspace = textscan(fid,'%f','headerlines',spacingline-1);
    gridspace = gridspace{1};

    y = [-128*gridspace:gridspace:127*gridspace];
    x = [-127*gridspace:gridspace:128*gridspace];
    [gx,gy] = meshgrid(x,y);

    pcoord = [-pxsz*kernelsz:pxsz:pxsz*kernelsz];
    [px,py] = meshgrid(pcoord,pcoord);

    resamplePSF = interp2(gx,gy,PSFlayer,px,py,'spline');
    PSF(:,:,i) = resamplePSF;
    i = i+1;

    
    
    drawnow
    imagesc(resamplePSF);
    
end

PSF = PSF./max(PSF,[],'all');

if iwrite
    for j = 1:i-1
        if j == 1
            % First slice:
            imwrite(uint16(65535.*PSF(:,:,j)),[outname,'.tiff'])
        else
            % Subsequent slices:
            imwrite(uint16(65535.*PSF(:,:,j)),[outname,'.tiff'],'WriteMode','append');
        end     
    end
end

save([outname,'.mat'],'PSF');