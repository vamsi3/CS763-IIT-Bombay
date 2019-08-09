function displayvideo (vid,pausetime)

T = size(vid,2);
for i=1:T
   imshow(vid(i).cdata);
   pause(pausetime);
end

