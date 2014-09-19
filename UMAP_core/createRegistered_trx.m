function out = createRegistered_trx(flyID,flymatrix,blobColor)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[roughtrx] = createTrajectory(flyID,flymatrix,blobColor);
[goodtrx] = selectgoodtrx(roughtrx);
[thetaPtoJpi] = convertThetaPtoJ(goodtrx);
[trx] = create_gpJAABAformat(goodtrx,thetaPtoJpi);
timestamps=(1:1:max(flymatrix(:,1))) ./30;

save registered_trx timestamps trx;
end

