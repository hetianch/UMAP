function [ goodtrxnew] = delete_duplicate_frame_goodtrx( goodtrx)
%this function is used to delete the duplicated frames
goodtrxnew=goodtrx;
for i= 1:3
    goodtrxnew(i).frame=goodtrx(i).frame(2:end);
      goodtrxnew(i).nFlies=goodtrx(i).nFlies(2:end);

        goodtrxnew(i).flyFrame=goodtrx(i).flyFrame(2:end);

          goodtrxnew(i).blobX=goodtrx(i).blobX(2:end);

            goodtrxnew(i).blobY=goodtrx(i).blobY(2:end);

              goodtrxnew(i).blobArea=goodtrx(i).blobArea(2:end);

                goodtrxnew(i).blobAngle=goodtrx(i).blobAngle(2:end);

                  goodtrxnew(i).blobA=goodtrx(i).blobA(2:end);

                    goodtrxnew(i).blobB=goodtrx(i).blobB(2:end);

                      goodtrxnew(i).blobColor=goodtrx(i).blobColor(2:end);

                        goodtrxnew(i).blobdeltaX=goodtrx(i).blobdeltaX(2:end);

                          goodtrxnew(i).blobdeltaY=goodtrx(i).blobdeltaY(2:end);



end

