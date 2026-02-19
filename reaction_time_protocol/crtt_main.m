%% SCRIPT FOR ADMINISTERING THE 2-CHOICE REACTION TIME TASK (CRTT)
% Revised version 01/2026. This script should be run before and after the
% TSST phase in order to analyse potential alterations in cognitive
% performance.

rng('shuffle')
format long;
clc;
clear all;
close all;

Info=inputdlg({'Path to save documents:','Name:'},'Input',1, {'C:\Users\','teste'});
name=strcat(Info{2},'.mat');
path=strcat(Info{1},'\');
nr_set=1;

Screen('Close'); sca;

Screen('Preference','Verbosity',0)
Screen('Preference','SkipSyncTests',1);
Screen('Preference','VisualDebugLevel',0);
KbName('UnifyKeyNames');
HideCursor;

screenNum=0;
flipSpd=1; % a flip every 13 frames
[wPtr,rect]=Screen('OpenWindow',screenNum);
monitorFlipInterval=Screen('GetFlipInterval', wPtr);
[X,Y] = RectCenter(rect);
resolucao=2*X;

kill=0;             % stop the stimuli if an error occurs

% blank the Screen and wait a second
Screen('FillRect',wPtr,[80 80 80]);
Screen(wPtr, 'Flip');
HideCursor;

tcal=GetSecs;
while GetSecs-tcal<1.5
    Screen('TextSize', wPtr, 50);
    Screen('DrawText', wPtr, 'Ready...', X-110, Y-45, [0 0 0]);
    vbl=Screen(wPtr, 'Flip');
end
WaitSecs(1);


i=0;
tstart=tic;

if nr_set==1
    respostas={}; %array of answers' ground truth
    respostas_corr={}; %array of participant's given answers
    instantes_int={}; %array of interstimulus intervals
    sequencia={}; %array detailing which one of the 4 stimuli is presented
    tempos_resp={}; %array of participant's answers response time
    tempos_estim={}; %array of time at which the stimulus is presented from the starting block (reset at every 200s)

end

set_number=num2str(nr_set);
set_number=strcat('_set',set_number);

while i<3 %Three blocks to save data along the way in order not to loose too much data in case of errors
    tstart=tic;
    resp=[];
    tempos=[];
    respcorr=[];
    seq=[];
    estim=[];

    %each block
    inst=[];
    tin=GetSecs;
    while toc(tstart)<200 %Three blocks of 200s --> 10min trial

        t1=GetSecs;

        b=(3-1).*rand(1)+(1); %generation of a random interstimulus interval between 0.5 and 3s
        inst=[inst b];
        rank=randperm(4);
        c=rank(1); %random number between 1 and 4 to select the stimulus
        seq=[seq c]; 
        nr_stim=length(seq);

        %Waiting three seconds before the first stimulus is shown
        while GetSecs-t1<3
            Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
        end

        tim=GetSecs;
        if (c==1)
            respcorr=[respcorr '1 ']; %1 for congruent stimuli
            t1=GetSecs;
            img=imread('S_S.png');
            img2=imresize(img,0.8);
            tex = Screen('MakeTexture', wPtr, double(img2));


            Screen('DrawTexture', wPtr, tex);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
            estim=[estim GetSecs-tin];
            ti = GetSecs;
            count=0;
            KbQueueCreate;
            KbQueueStart;

            %if the stimulus response is lower than 100ms then invalidate
            %the the answers as no distinction process has occurred 10.1037/0022-3514.85.2.197
            while GetSecs-ti<0.3 

                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    % pressedCodes=find(firstPress);
                    % for j=1:size(pressedCodes,2)
                    %     key = KbName(pressedCodes(j));
                    %     secs=firstPress(pressedCodes(j))-ti;
                    % end
                    resp{nr_stim}(1,count)=0;
                    tempos{nr_stim}(1,count)=-1;
                end
            end

            Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));

            while (GetSecs-ti<b)
                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    pressedCodes=find(firstPress);
                    for k=1:size(pressedCodes,2)
                        key = KbName(pressedCodes(k));
                        secs=firstPress(pressedCodes(k))-ti;
                    end

                    if strcmp(key(1),'o')
                        resp{nr_stim}(1,count)='o';

                        for l=1:length(resp)
                            if isempty(resp{l})==1
                                resp{l}=0;
                                tempos{l}=-1;
                            end
                        end

                        save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                        Screen('CloseAll');
                        ShowCursor;
                        return;
                    end

                    if isempty(key(1))==1
                        resp{nr_stim}(1,count)=0;
                        tempos{nr_stim}(1,count)=-1;
                    else
                        resp{nr_stim}(1,count)=key(1);
                        tempos{nr_stim}(1,count)=secs;
                    end

                end

                if kill==1
                    break;
                end

            end

            KbQueueRelease

            for r=1:length(resp)
                if isempty(resp{r})==1
                    resp{r}=0;
                    tempos{r}=-1;
                end
            end

        elseif (c==2)

            respcorr=[respcorr '3 ']; %3 for incongruent stimuli
            t1=GetSecs;
            img=imread('S_H.png');
            img2=imresize(img,0.8);
            tex = Screen('MakeTexture', wPtr, double(img2));

            Screen('DrawTexture', wPtr, tex);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
            estim=[estim GetSecs-tin];
            ti = GetSecs;
            count=0;
            KbQueueCreate;
            KbQueueStart;

            while GetSecs-ti<0.3

                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    % pressedCodes=find(firstPress);
                    % for v=1:size(pressedCodes,2)
                    %     key = KbName(pressedCodes(v))
                    %     secs=firstPress(pressedCodes(v))-ti;
                    % end
                    resp{nr_stim}(1,count)=0;
                    tempos{nr_stim}(1,count)=-1;
                end
            end

            Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));

            while (GetSecs-ti<b)
                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    pressedCodes=find(firstPress);
                    for h=1:size(pressedCodes,2)
                        key = KbName(pressedCodes(h));
                        secs=firstPress(pressedCodes(h))-ti;
                    end

                    if strcmp(key(1),'o')
                        resp{nr_stim}(1,count)='o';

                        for w=1:length(resp)
                            if isempty(resp{w})==1
                                resp{w}=0;
                                tempos{w}=-1;
                            end
                        end

                        save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                        Screen('CloseAll');
                        ShowCursor;
                        return;
                    end

                    if isempty(key(1))==1
                        resp{nr_stim}(1,count)=0;
                        tempos{nr_stim}(1,count)=-1;
                    else
                        resp{nr_stim}(1,count)=key(1);
                        tempos{nr_stim}(1,count)=secs;
                    end

                end

                if kill==1
                    break;
                end

            end


            KbQueueRelease

            for z=1:length(resp)
                if isempty(resp{z})==1
                    resp{z}=0;
                    tempos{z}=-1;
                end
            end

        elseif (c==3)
            respcorr=[respcorr '3 '];
            t1=GetSecs;
            img=imread('H_S.png');
            img2=imresize(img,0.8);
            tex = Screen('MakeTexture', wPtr, double(img2));

            Screen('DrawTexture', wPtr, tex);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
            estim=[estim GetSecs-tin];
            ti = GetSecs;
            count=0;
            KbQueueCreate;
            KbQueueStart;

            while GetSecs-ti<0.3

                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    % pressedCodes=find(firstPress);
                    % for j=1:size(pressedCodes,2)
                    %     key = KbName(pressedCodes(j))
                    %     secs=firstPress(pressedCodes(j))-ti;
                    % end
                    resp{nr_stim}(1,count)=0;
                    tempos{nr_stim}(1,count)=-1;
                end
            end

            Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));

            while (GetSecs-ti<b)
                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    pressedCodes=find(firstPress);
                    for k=1:size(pressedCodes,2)
                        key = KbName(pressedCodes(k));
                        secs=firstPress(pressedCodes(k))-ti;
                    end

                    if strcmp(key(1),'o')
                        resp{nr_stim}(1,count)='o';

                        for l=1:length(resp)
                            if isempty(resp{l})==1
                                resp{l}=0;
                                tempos{l}=-1;
                            end
                        end

                        save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                        Screen('CloseAll');
                        ShowCursor;
                        return;
                    end

                    if isempty(key(1))==1
                        resp{nr_stim}(1,count)=0;
                        tempos{nr_stim}(1,count)=-1;
                    else
                        resp{nr_stim}(1,count)=key(1);
                        tempos{nr_stim}(1,count)=secs;
                    end

                end

                if kill==1
                    break;
                end

            end

            KbQueueRelease

            for r=1:length(resp)
                if isempty(resp{r})==1
                    resp{r}=0;
                    tempos{r}=-1;
                end
            end

        elseif (c==4)
            respcorr=[respcorr '1 '];
            t1=GetSecs;
            img=imread('H_H.png');
            img2=imresize(img,0.8);
            tex = Screen('MakeTexture', wPtr, double(img2));


            Screen('DrawTexture', wPtr, tex);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
            estim=[estim GetSecs-tin];
            ti = GetSecs;
            count=0;
            KbQueueCreate;
            KbQueueStart;

            while GetSecs-ti<0.3

                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    % pressedCodes=find(firstPress);
                    % for j=1:size(pressedCodes,2)
                    %     key = KbName(pressedCodes(j))
                    %     secs=firstPress(pressedCodes(j))-ti;
                    % end
                    resp{nr_stim}(1,count)=0;
                    tempos{nr_stim}(1,count)=-1;
                end
            end

            Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
            vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));

            while (GetSecs-ti<b)
                [keyIsDown, firstPress] = KbQueueCheck;
                if keyIsDown==1

                    count=count+1;
                    pressedCodes=find(firstPress);
                    for k=1:size(pressedCodes,2)
                        key = KbName(pressedCodes(k));
                        secs=firstPress(pressedCodes(k))-ti;
                    end

                    if strcmp(key(1),'o')
                        resp{nr_stim}(1,count)='o';

                        for l=1:length(resp)
                            if isempty(resp{l})==1
                                resp{l}=0;
                                tempos{l}=-1;
                            end
                        end

                        save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                        Screen('CloseAll');
                        ShowCursor;
                        return;
                    end

                    if isempty(key(1))==1
                        resp{nr_stim}(1,count)=0;
                        tempos{nr_stim}(1,count)=-1;
                    else
                        resp{nr_stim}(1,count)=key(1);
                        tempos{nr_stim}(1,count)=secs;
                    end

                end

                if kill==1
                    break;
                end

            end

            KbQueueRelease

            for r=1:length(resp)
                if isempty(resp{r})==1
                    resp{r}=0;
                    tempos{r}=-1;
                end
            end
        end
    end

    i=i+1;
    if nr_set==1
        instantes_int{i}=inst;
        sequencia{i}=seq;
        respostas{i}=resp;
        respostas_corr{i}=respcorr;
        tempos_resp{i}=tempos;
        tempos_estim{i}=estim;
    else
        instantes_int{3*(nr_set-1)+i}=inst;
        sequencia{3*(nr_set-1)+i}=seq;
        respostas{3*(nr_set-1)+i}=resp;
        respostas_corr{3*(nr_set-1)+i}=respcorr;
        tempos_resp{3*(nr_set-1)+i}=tempos;
        tempos_estim{3*(nr_set-1)+i}=estim;
    end
end

%Casting to double the values saved in single cells for better handling
for i = 1:length(respostas)
    tempos_resp{1,i}=cell2mat(tempos_resp{1,i});
    respostas{1,i}=str2double(string(respostas{1,i}));
    respostas_corr{1,i} = str2num(respostas_corr{1,i});
end

Screen('CloseAll');
ShowCursor

% In case there is a discrepancy between the size of respostas/tempos_resp
% and the other ground truth variables (e.g., because answers missed
% occurred between two consecutive blocks), the following for cycle corrects for it

for i = 1:length(respostas)
    for j = 1:length(tempos_estim{i})
        if length(tempos_estim{i}(1:j))>length(respostas{i})
            respostas{i}(end+1) = 0;
            tempos_resp{i}(end+1) = -1;
        end
    end
end

clearvars -except instantes_int respostas respostas_corr sequencia tempos_estim tempos_resp
save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int', 'tempos_estim')

%% RE-ORGANISATION
% Two .mat files are created for CRTT1 and CRTT2 that must be concatenated,
% then the size of each is computed in order to understand the boundaries
% of the separate trials.

crtt1 = importdata('teste2.mat'); 
crtt2 = importdata('teste1.mat');

sizeT1 = length([crtt1.instantes_int{:}]); sizeT2 = length([crtt2.instantes_int{:}]);

instantes_int = [crtt1.instantes_int{:}, crtt2.instantes_int{:}]; 
respostas = [crtt1.respostas{:}, crtt2.respostas{:}]; 
respostas_corr = [crtt1.respostas_corr{:}, crtt2.respostas_corr{:}];
sequencia = [crtt1.sequencia{:}, crtt2.sequencia{:}];
tempos_estim = [crtt1.tempos_estim{:}, crtt2.tempos_estim{:}];
tempos_resp = [crtt1.tempos_resp{:}, crtt2.tempos_resp{:}];

save('Teste_A','instantes_int','respostas','respostas_corr','sequencia','tempos_estim','tempos_resp','sizeT1','sizeT2')
