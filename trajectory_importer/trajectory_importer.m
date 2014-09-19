function varargout =trajectory_importer(varargin)
%trajectory_importer MATLAB code fortrajectory_importer.fig
%     trajectory_importer, by itself, creates a newtrajectory_importer or raises the existing
%      singleton*.
%
%      H =trajectory_importer returns the handle to a newtrajectory_importer or the handle to
%      the existing singleton*.
%
%     trajectory_importer('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK intrajectory_importer.M with the given input arguments.
%
%     trajectory_importer('Property','Value',...) creates a newtrajectory_importer or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI beforetrajectory_importer_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed totrajectory_importer_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to helptrajectory_importer

% Last Modified by GUIDE v2.5 10-Jul-2014 16:46:03

% Begin initialization code - DO NOT EDIT
% see Callbacks for Specific Components for instructions
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @trajectory_importer_OpeningFcn, ...
                   'gui_OutputFcn',  @trajectory_importer_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just beforetrajectory_importer is made visible.
function trajectory_importer_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments totrajectory_importer (see VARARGIN)

% Choose default command line output fortrajectory_importer
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makestrajectory_importer wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout =trajectory_importer_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes when figure1 is resized.
function figure1_ResizeFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function path_Callback(hObject, eventdata, handles)
% hObject    handle to path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of path as text
%        str2double(get(hObject,'String')) returns contents of path as a double


% --- Executes during object creation, after setting all properties.
function path_CreateFcn(hObject, eventdata, handles)
% hObject    handle to path (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in frame_diagnosis.
function frame_diagnosis_Callback(hObject, eventdata, handles)
% hObject    handle to frame_diagnosis (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of frame_diagnosis


% --- Executes on button press in load.
function load_Callback(hObject, eventdata, handles)
% hObject    handle to load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
path=nan;
frame_diagnosis=nan;
center_x=nan;
center_y=nan;
arena_diameter=nan;
frames_second=nan;
pixels_mm=nan;

time_start=fix(clock);
path=get(handles.path,'String');
frame_diagnosis=get(handles.frame_diagnosis,'Value');
center_x=str2double(get(handles.center_x,'String'));
center_y=str2double(get(handles.center_y,'String'));
arena_diameter=str2double(get(handles.arena_diameter,'String'));
frames_second=str2double(get(handles.frames_second,'String'));
pixels_mm=str2double(get(handles.pixels_mm,'String'));



if (isnan(center_x)|isnan(center_y)|isnan(arena_diameter)|isnan(frames_second)|isnan(frames_second))
    set(handles.monitor,'String','Please input numeric values for all parameters in Conversion panel');    
elseif isempty(path)
    set(handles.monitor,'String','no input file found');
else
    set(handles.monitor,'String','generating trajectory for JAABA...');
    drawnow;
    start_time=tic;
    [flyID, blobColor ,flymatrix]=load_trajectory(path);
    [goodtrx, trx] = create_JAABA_trajectory(flyID,flymatrix,blobColor,center_x,center_y,arena_diameter,frames_second,pixels_mm)
    clear flyID flymatrix;

     if frame_diagnosis==0
        time_eclipse=toc(start_time);
        show_string=strcat('trajectory for JAABA has been generated successfully. Runing time: ', num2str(time_eclipse),'s');
        set(handles.monitor,'String',show_string);
        drawnow;
     else   
        flynumber = size (goodtrx,2);
        error=0;
        fly_idx=[];
        frame_idx=[];

        for i =1:flynumber
             framenumber= size(goodtrx(i).frame,1) ;
             fly_idx_temp=[];
             frame_idx_temp=[];
             for j = 2: framenumber
                 if goodtrx(i).frame(j) ~=goodtrx(i).frame(j-1)+1
                    error=1;
                    fly_idx_temp=i;
                    frame_idx_temp=j;
                    fly_idx=[fly_idx;fly_idx_temp];
                   frame_idx=[frame_idx;frame_idx_temp];   
                end 
             end
        end


       if error==0
          time_eclipse=toc(start_time);
          show_string=strcat('no inconsecutive frame found. Runing time: ', num2str(time_eclipse),'s');
          set(handles.monitor,'String',show_string);
          drawnow;
       else
          time_eclipse=toc(start_time);
          show_string=strcat('no inconsecutive frame found. Runing time: ', num2str(time_eclipse),'s');
          set(handles.monitor,'String',show_string);
          drawnow;
          time=fix(clock);
          time_string=strcat(num2str(time(1)),num2str(time(2)),num2str(time(3)),num2str(time(4)),num2str(time(5)));
          errorlog_name=strcat('errorlog',time_string);
          diary(errorlog_name);
             for i=1:length(fly_idx);
                 fprintf('find inconsecutive frame for fly %s at frame %f\n', goodtrx(fly_idx(i)).originalIdx{1},goodtrx(fly_idx(i)).frame(frame_idx(i)));
             end
             diary off;
        end 
     end
 
end

% --- Executes on button press in clear.
function clear_Callback(hObject, eventdata, handles)
% hObject    handle to clear (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(handles.center_x,'String','');
set(handles.center_y,'String','');
set(handles.frame_diagnosis,'Value',0);
set(handles.path,'String','');
set(handles.arena_diameter,'String','');
set(handles.frames_second,'String','');
set(handles.pixels_mm,'String','');
set(handles.monitor,'String','Please input numeric value into Conversion panel');



% --- Executes on button press in quit.
function quit_Callback(hObject, eventdata, handles)
% hObject    handle to quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get the current position of the GUI from the handles structure
% to pass to the modal dialog.
close all;

function center_x_Callback(hObject, eventdata, handles)
% hObject    handle to center_x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of center_x as text
%        str2double(get(hObject,'String')) returns contents of center_x as a double


% --- Executes during object creation, after setting all properties.
function center_x_CreateFcn(hObject, eventdata, handles)
% hObject    handle to center_x (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function arena_diameter_Callback(hObject, eventdata, handles)
% hObject    handle to arena_diameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of arena_diameter as text
%        str2double(get(hObject,'String')) returns contents of arena_diameter as a double


% --- Executes during object creation, after setting all properties.
function arena_diameter_CreateFcn(hObject, eventdata, handles)
% hObject    handle to arena_diameter (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function center_y_Callback(hObject, eventdata, handles)
% hObject    handle to center_y (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of center_y as text
%        str2double(get(hObject,'String')) returns contents of center_y as a double


% --- Executes during object creation, after setting all properties.
function center_y_CreateFcn(hObject, eventdata, handles)
% hObject    handle to center_y (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function frames_second_Callback(hObject, eventdata, handles)
% hObject    handle to frames_second (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of frames_second as text
%        str2double(get(hObject,'String')) returns contents of frames_second as a double


% --- Executes during object creation, after setting all properties.
function frames_second_CreateFcn(hObject, eventdata, handles)
% hObject    handle to frames_second (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



function pixels_mm_Callback(hObject, eventdata, handles)
% hObject    handle to pixels_mm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: get(hObject,'String') returns contents of pixels_mm as text
%        str2double(get(hObject,'String')) returns contents of pixels_mm as a double


% --- Executes during object creation, after setting all properties.
function pixels_mm_CreateFcn(hObject, eventdata, handles)
% hObject    handle to pixels_mm (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: edit controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end

% --- Executes on button press in help_push.
function help_push_Callback(hObject, eventdata, handles)
% hObject    handle to help_push (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
button_state=get(hObject,'Value');
if button_state==get(hObject,'Max')
    set(handles.help_panel,'Visible','on');
elseif button_state==get(hObject,'Min')
    set(handles.help_panel,'Visible','off');
end
