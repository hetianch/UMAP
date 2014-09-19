function [trxnew] = delete_duplicate_frame_trx(trx)
%this function is used to delete the duplicated frames
trxnew=trx;
for i= 1:3
    trxnew(i).x=trx(i).x(2:end);
      trxnew(i).y=trx(i).y(2:end);
      trxnew(i).x_mm=trx(i).x_mm(2:end);
      trxnew(i).y_mm=trx(i).y_mm(2:end);


        trxnew(i).theta=trx(i).theta(2:end);
         trxnew(i).theta_mm=trx(i).theta_mm(2:end);

          trxnew(i).a=trx(i).a(2:end);
          trxnew(i).b=trx(i).b(2:end);
          trxnew(i).a_mm=trx(i).a_mm(2:end);
          trxnew(i).b_mm=trx(i).b_mm(2:end);
trxnew(i).nframes=size(trxnew(i).x,2);
trxnew(i).endframe= trxnew(i).nframes+trx(i).firstframe-1;

end

