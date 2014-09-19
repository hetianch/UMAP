function [ changeW ] = computeChangeWindow(params,x )
ROW=size(x,1);
changeW =[];
change_radii=params.window.change.change_window_radii;
radii=params.window.window_radius;

for change_radius=change_radii
    changeW_radi_temp=[];
    valid_radii=radii(find(2*(2*change_radius+1)<2*radii+1));

    if isempty(valid_radii)==1
        changeW_radi_temp=[];
    else


        for windowRadius=valid_radii
            changeW_temp=[];

            if windowRadius==0
               changeW_temp=x;
            else 
               changeW_temp=[];
               changeW_temp1=[];
               changeW_temp2=[];
               changeW_temp3=[];

                    if isempty(find(params.window.window_offset==-1))==1
                        changeW_temp1=[];
                    else 
                        for i=1:ROW
                            if i< 2*windowRadius+1
                                 changeW_temp1(i)=NaN;
                            else
                                 changeW_temp1(i)=nanmean ( x((i-2*change_radius):i)) - nanmean( x( (i-2*windowRadius): (i-2*windowRadius+2*2*change_radius) ));

                            end
                        end
                    end    

                    if isempty(find(params.window.window_offset==0))==1
                        changeW_temp2=[];
                    else     
                        for i=1:ROW
                             if (i<windowRadius+1) | (ROW-i<windowRadius) 
                             changeW_temp2(i)=NaN;
                             else
                             changeW_temp2(i)= nanmean(x( (i+windowRadius-2*change_radius):(i+windowRadius)  ) )-nanmean( x( (i-windowRadius): (i-windowRadius+2*change_radius) ) );

                             end
                        end
                    end

                    if isempty(find(params.window.window_offset==1))==1
                        changeW_temp3=[]; 
                    else
                        for i=1:ROW
                            if ROW-i<2*windowRadius
                                 changeW_temp3(i)=NaN;
                            else
                             changeW_temp3(i)= nanmean(  x( (i+2*windowRadius-2*change_radius): (i+2*windowRadius) ) )-nanmean( x(i:(i+2*change_radius)));
                            end
                        end
                    end
            changeW_temp=[changeW_temp1',changeW_temp2',changeW_temp3'];            
            end
            changeW_radi_temp=[changeW_radi_temp,changeW_temp]; 
        end
    changeW =[changeW,changeW_radi_temp];
    end
end






