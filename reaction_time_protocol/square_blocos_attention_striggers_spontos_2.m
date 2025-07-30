rng('shuffle')
format long;

Nr_set=inputdlg('Set number (1,2,3): ','Tarefa Square Grating');
nr_set=str2double(Nr_set{1});

if nr_set==1
    clc;
    clear all;
    close all;
    Info=inputdlg({'Path to save documents:','Name:'},'Input',1, {'C:\Users\','teste'});
    name=strcat(Info{2},'.mat');
    path=strcat(Info{1},'\');
    nr_set=1;
else
    load(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int', 'tempos_estim')
end

Screen('Close'); sca;


%try

%     app=[0.0005, 0.00002, 0.1, 0.0002, 0.05];

    app=[0.15, 0.1, 0.3, 0.5];


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
        respostas={};
        respostas_corr={};
        instantes_int={};
        sequencia={};
        tempos_resp={};
        tempos_estim={};

    end


    set_number=num2str(nr_set);
    set_number=strcat('_set',set_number);
    
    while i<3
        tstart=tic;
        resp=[];
        tempos=[];
        respcorr=[];
        seq=[];
        estim=[];

%         %ponto de fixação a aparecer entre blocos durante 12 seg.
%         Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%         vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
%         b=15;
%         delta_t = 0;
%         ti=GetSecs;
%         
%             
%         while (delta_t<b)
%             delta_t=GetSecs-ti;
%         end

            %cada bloco
            inst=[];
            tin=GetSecs;
            while toc(tstart)<240
    %             count=count+1;
                t1=GetSecs;

                b=(3-1).*rand(1)+(1); %gera nr aleatorio entre 0.5 e 10
                inst=[inst b];
                rank=randperm(4);
                c=rank(1); %nr aleatorio entre 1 e 2 
                seq=[seq c]; %para gravar que seta apareceu 
                nr_stim=length(seq);
                
                    
                 %esperar 3 seg antes de aparecer a seta1
                while GetSecs-t1<3
%                     img=imread('squarewithouttriangle.bmp');
%                     img2=imresize(img,0.1);
%                     tex = Screen('MakeTexture', wPtr, double(img2));
                    Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%                     Screen('DrawTexture', wPtr, tex);
                    vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));
                end

                tim=GetSecs;
                if (c==1)
                    respcorr=[respcorr 'n '];
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
                    
                    p=randperm(length(app));
                    app(p(1))

                    while GetSecs-ti<app(p(1))

                        [keyIsDown, firstPress] = KbQueueCheck;
                        if keyIsDown==1


                            count=count+1;
                            pressedCodes=find(firstPress);
                            for j=1:size(pressedCodes,2)
                                key = KbName(pressedCodes(j));
                                secs=firstPress(pressedCodes(j))-ti;
                            end

                            resp{nr_stim}(1,count)=key(1);
                            tempos{nr_stim}(1,count)=secs-ti;
                        end
                    end

%                     img=imread('squarewithouttriangle.bmp');
%                     img2=imresize(img,0.1);
%                     tex = Screen('MakeTexture', wPtr, double(img2));
                    Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%                     Screen('DrawTexture', wPtr, tex);
                    vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));


                    while (GetSecs-ti<b)
                        [keyIsDown, firstPress] = KbQueueCheck;
                            if keyIsDown==1

                                count=count+1;
                                pressedCodes=find(firstPress);
                                for k=1:size(pressedCodes,2)
                                    key = KbName(pressedCodes(k))
                                    secs=firstPress(pressedCodes(k))-ti;
                                end

                                if strcmp(key(1),'o')
                                    resp{nr_stim}(1,count)='o';

                                    for l=1:length(resp)
                                        if isempty(resp{l})==1
                                        resp{l}='-';
                                        tempos{l}=-1;
                                        end
                                    end
                                    
                                    save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                                    Screen('CloseAll');
                                    ShowCursor;
                                    return;
                                end

                              if isempty(key(1))==1
                                  resp{nr_stim}(1,count)='-';
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
                            resp{r}='-';
                            tempos{r}=-1;
                        end
                    end

                   elseif (c==2)  

                        respcorr=[respcorr 'm '];
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
                        
                        p=randperm(length(app));
                        app(p(1))
                        
                        while GetSecs-ti<app(p(1))

                            [keyIsDown, firstPress] = KbQueueCheck;
                            if keyIsDown==1

                                count=count+1;
                                pressedCodes=find(firstPress);
                                for v=1:size(pressedCodes,2)
                                    key = KbName(pressedCodes(v))
                                    secs=firstPress(pressedCodes(v))-ti;
                                end

                                resp{nr_stim}(1,count)=key(1);
                                tempos{nr_stim}(1,count)=secs-ti;
                            end
                        end

%                         img=imread('squarewithouttriangle.bmp');
%                         img2=imresize(img,0.1);
%                         tex = Screen('MankeTexture', wPtr, double(img2));
                        Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%                         Screen('DrawTexture', wPtr, tex);
                        vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));


                        while (GetSecs-ti<b)
                            [keyIsDown, firstPress] = KbQueueCheck;
                                if keyIsDown==1

                                    count=count+1;
                                    pressedCodes=find(firstPress);
                                    for h=1:size(pressedCodes,2)
                                        key = KbName(pressedCodes(h))
                                        secs=firstPress(pressedCodes(h))-ti;
                                    end

                                    if strcmp(key(1),'o')
                                        resp{nr_stim}(1,count)='o';

                                        for w=1:length(resp)
                                            if isempty(resp{w})==1
                                            resp{w}='-';
                                            tempos{w}=-1;
                                            end
                                        end

                                        save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                                        Screen('CloseAll');
                                        ShowCursor;
                                        return;
                                    end

                                  if isempty(key(1))==1
                                      resp{nr_stim}(1,count)='-';
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
                                resp{z}='-';
                                tempos{z}=-1;
                            end
                        end
                        
                elseif (c==3)
                    respcorr=[respcorr 'm '];
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
                    
                    p=randperm(length(app));
                    app(p(1))

                    while GetSecs-ti<app(p(1))

                        [keyIsDown, firstPress] = KbQueueCheck;
                        if keyIsDown==1


                            count=count+1;
                            pressedCodes=find(firstPress);
                            for j=1:size(pressedCodes,2)
                                key = KbName(pressedCodes(j))
                                secs=firstPress(pressedCodes(j))-ti;
                            end

                            resp{nr_stim}(1,count)=key(1);
                            tempos{nr_stim}(1,count)=secs-ti;
                        end
                    end

%                     img=imread('squarewithouttriangle.bmp');
%                     img2=imresize(img,0.1);
%                     tex = Screen('MakeTexture', wPtr, double(img2));
                    Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%                     Screen('DrawTexture', wPtr, tex);
                    vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));


                    while (GetSecs-ti<b)
                        [keyIsDown, firstPress] = KbQueueCheck;
                            if keyIsDown==1

                                count=count+1;
                                pressedCodes=find(firstPress);
                                for k=1:size(pressedCodes,2)
                                    key = KbName(pressedCodes(k))
                                    secs=firstPress(pressedCodes(k))-ti;
                                end

                                if strcmp(key(1),'o')
                                    resp{nr_stim}(1,count)='o';

                                    for l=1:length(resp)
                                        if isempty(resp{l})==1
                                        resp{l}='-';
                                        tempos{l}=-1;
                                        end
                                    end
                                    
                                    save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                                    Screen('CloseAll');
                                    ShowCursor;
                                    return;
                                end

                              if isempty(key(1))==1
                                  resp{nr_stim}(1,count)='-';
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
                            resp{r}='-';
                            tempos{r}=-1;
                        end
                    end
                 elseif (c==4)
                    respcorr=[respcorr 'n '];
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
                    
                    p=randperm(length(app));
                    app(p(1))

                    while GetSecs-ti<app(p(1))

                        [keyIsDown, firstPress] = KbQueueCheck;
                        if keyIsDown==1


                            count=count+1;
                            pressedCodes=find(firstPress);
                            for j=1:size(pressedCodes,2)
                                key = KbName(pressedCodes(j))
                                secs=firstPress(pressedCodes(j))-ti;
                            end

                            resp{nr_stim}(1,count)=key(1);
                            tempos{nr_stim}(1,count)=secs-ti;
                        end
                    end

%                     img=imread('squarewithouttriangle.bmp');
%                     img2=imresize(img,0.1);
%                     tex = Screen('MakeTexture', wPtr, double(img2));
                    Screen('DrawDots', wPtr, [X;Y], 15, [230, 230, 230], [0 0], 1);
%                     Screen('DrawTexture', wPtr, tex);
                    vbl=Screen(wPtr, 'Flip', vbl+(flipSpd*monitorFlipInterval));


                    while (GetSecs-ti<b)
                        [keyIsDown, firstPress] = KbQueueCheck;
                            if keyIsDown==1

                                count=count+1;
                                pressedCodes=find(firstPress);
                                for k=1:size(pressedCodes,2)
                                    key = KbName(pressedCodes(k))
                                    secs=firstPress(pressedCodes(k))-ti;
                                end

                                if strcmp(key(1),'o')
                                    resp{nr_stim}(1,count)='o';

                                    for l=1:length(resp)
                                        if isempty(resp{l})==1
                                        resp{l}='-';
                                        tempos{l}=-1;
                                        end
                                    end
                                    
                                    save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int','tempos_estim')
                                    Screen('CloseAll');
                                    ShowCursor;
                                    return;
                                end

                              if isempty(key(1))==1
                                  resp{nr_stim}(1,count)='-';
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
                            resp{r}='-';
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


    save(strcat(path,name),'respostas','respostas_corr','sequencia','tempos_resp', 'instantes_int', 'tempos_estim')
    
    Screen('CloseAll');
    ShowCursor
    
%catch

%     Screen('CloseAll');
%     ShowCursor;
    
%end

