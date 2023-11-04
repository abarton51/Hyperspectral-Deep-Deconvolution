function [gknext] = RLcore(OTF,IOTF,O,outvol,iters)
    gknext = gpuArray(ones(outvol,'single'));
    fwdnext = O;

    for i = 1:iters
        [gknext,fwdnext] = RLiter(gknext,OTF,IOTF,O,fwdnext);
    end

end

function [gknext,fwdnext] = RLiter(gk,OTF,IOTF,O,fwdproj)
    bkproj = real(ifft2(IOTF.*fft2(fwdproj)));
    gknext = gk.*bkproj;
    gknext(isnan(gknext)) = 0;
    %gknext = gknext + max(gknext,[],'all')*0.001; %numerical offset factor
    
    fwdnext = O./sum(real(ifft2(OTF.*fft2(gknext))),3);
    fwdnext(isnan(fwdnext)) = 0;
    %fwdnext = fwdnext + max(fwdnext,[],'all').*0.001;
end
