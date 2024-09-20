function [x,y]=get_cursor_value(image,title_name)
%     img=dicomread(image);
    fig=figure;
    imshow(image,[])
    title(title_name);
    title(title_name,'Interpreter','none')
    set(gcf,'position',[0,0,1300,1300])
    % Enable data cursor mode
    datacursormode on
    dcm_obj = datacursormode(fig);
    % Set update function
    set(dcm_obj,'UpdateFcn',@myupdatefcn)
    % Wait while the user to click
    disp('Click line to display a data tip, then press "Return"')
    pause 
    % Export cursor to workspace
    info_struct = getCursorInfo(dcm_obj);
    if isfield(info_struct, 'Position')
      disp('Clicked positioin is')
      disp(info_struct.Position)
      x=info_struct.Position(1);
      y=info_struct.Position(2);
    end
    close all
    function output_txt = myupdatefcn(~,event_obj)
      % ~            Currently not used (empty)
      % event_obj    Object containing event data structure
      % output_txt   Data cursor text
      pos = get(event_obj, 'Position');
      output_txt = {['x: ' num2str(pos(1))], ['y: ' num2str(pos(2))]};
    end
end