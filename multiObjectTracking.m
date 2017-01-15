function multiObjectTracking()

% create system objects used for reading video, detecting moving objects,
% and displaying the results
obj = setupSystemObjects(); %初始化函数
tracks = initializeTracks(); % create an empty array of tracks  %初始化轨迹对象

nextId = 1; % ID of the next track

% detect moving objects, and track them across video frames
while ~isDone(obj.reader)
    frame = readFrame();  %读取一帧
    [centroids, bboxes, mask] = detectObjects(frame); %前景检测
    predictNewLocationsOfTracks();  %根据位置进行卡尔曼预测
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment(); %匈牙利匹配算法进行匹配
    
    updateAssignedTracks();%分配好的轨迹更新
    updateUnassignedTracks();%未分配的轨迹更新
    deleteLostTracks();%删除丢掉的轨迹
    createNewTracks();%创建新轨迹
    
    displayTrackingResults();%结果展示
end


%% Create System Objects
% Create System objects used for reading the video frames, detecting
% foreground objects, and displaying results.

    function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.
        
        % create a video file reader
    
        obj.reader = vision.VideoFileReader('001.wmv');         %读入视频
        
        % create two video players, one to display the video,
        % and one to display the foreground mask
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 400, 700, 400]);   %创建两个窗口
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        
        % Create system objects for foreground detection and blob analysis
        
        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background. 
        
        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...   %GMM进行前景检测，高斯核数目为3，前40帧为背景帧，域值为0.7
            'NumTrainingFrames', 200, 'MinimumBackgroundRatio', 0.7);   
        
        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis system object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.
        
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...  %输出质心和外接矩形
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 40,'MaximumBlobArea',100000);
    end

%% Initialize Tracks
% The |initializeTracks| function creates an array of tracks, where each
% track is a structure representing a moving object in the video. The
% purpose of the structure is to maintain the state of a tracked object.
% The state consists of information used for detection to track assignment,
% track termination, and display. 
%
% The structure contains the following fields:
%
% * |id| :                  the integer ID of the track
% * |bbox| :                the current bounding box of the object; used
%                           for display
% * |kalmanFilter| :        a Kalman filter object used for motion-based
%                           tracking
% * |age| :                 the number of frames since the track was first
%                           detected
% * |totalVisibleCount| :   the total number of frames in which the track
%                           was detected (visible)
% * |consecutiveInvisibleCount| : the number of consecutive frames for 
%                                  which the track was not detected (invisible).
%
% Noisy detections tend to result in short-lived tracks. For this reason,
% the example only displays an object after it was tracked for some number
% of frames. This happens when |totalVisibleCount| exceeds a specified 
% threshold.    
%
% When no detections are associated with a track for several consecutive
% frames, the example assumes that the object has left the field of view 
% and deletes the track. This happens when |consecutiveInvisibleCount|
% exceeds a specified threshold. A track may also get deleted as noise if 
% it was tracked for a short time, and marked invisible for most of the of 
% the frames.        

    function tracks = initializeTracks()
        % create an empty array of tracks
        tracks = struct(...
            'id', {}, ...  %轨迹ID
            'bbox', {}, ... %外接矩形
            'kalmanFilter', {}, ...%轨迹的卡尔曼滤波器
            'age', {}, ...%总数量
            'totalVisibleCount', {}, ...%可视数量
            'consecutiveInvisibleCount', {});%不可视数量
    end

%% Read a Video Frame
% Read the next video frame from the video file.
    function frame = readFrame()
        frame = obj.reader.step();%激活读图函数
    end

%% Detect Objects
% The |detectObjects| function returns the centroids and the bounding boxes
% of the detected objects. It also returns the binary mask, which has the 
% same size as the input frame. Pixels with a value of 1 correspond to the
% foreground, and pixels with a value of 0 correspond to the background.   
%
% The function performs motion segmentation using the foreground detector. 
% It then performs morphological operations on the resulting binary mask to
% remove noisy pixels and to fill the holes in the remaining blobs.  

    function [centroids, bboxes, mask] = detectObjects(frame)
        
        % detect foreground
        mask = step(obj.detector,frame);
        
        % apply morphological operations to remove noise and fill in holes
        mask = imopen(mask, strel('rectangle', [3,3]));%开运算
        mask = imclose(mask, strel('rectangle', [15, 15])); %闭运算

        mask = imfill(mask, 'holes');%填洞
        
        % perform blob analysis to find connected components
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
    end

%% Predict New Locations of Existing Tracks
% Use the Kalman filter to predict the centroid of each track in the
% current frame, and update its bounding box accordingly.

    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            
            % predict the current location of the track
            predictedCentroid = predict(tracks(i).kalmanFilter);%根据以前的轨迹，预测当前位置
            
            % shift the bounding box so that its center is at 
            % the predicted location
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];%真正的当前位置
        end
    end

%% Assign Detections to Tracks
% Assigning object detections in the current frame to existing tracks is
% done by minimizing cost. The cost is defined as the negative
% log-likelihood of a detection corresponding to a track.  
%
% The algorithm involves two steps: 
%
% Step 1: Compute the cost of assigning every detection to each track using
% the |distance| method of the |vision.KalmanFilter| System object. The 
% cost takes into account the Euclidean distance between the predicted
% centroid of the track and the centroid of the detection. It also includes
% the confidence of the prediction, which is maintained by the Kalman
% filter. The results are stored in an MxN matrix, where M is the number of
% tracks, and N is the number of detections.   
%
% Step 2: Solve the assignment problem represented by the cost matrix using
% the |assignDetectionsToTracks| function. The function takes the cost 
% matrix and the cost of not assigning any detections to a track.  
%
% The value for the cost of not assigning a detection to a track depends on
% the range of values returned by the |distance| method of the 
% |vision.KalmanFilter|. This value must be tuned experimentally. Setting 
% it too low increases the likelihood of creating a new track, and may
% result in track fragmentation. Setting it too high may result in a single 
% track corresponding to a series of separate moving objects.   
%
% The |assignDetectionsToTracks| function uses the Munkres' version of the
% Hungarian algorithm to compute an assignment which minimizes the total
% cost. It returns an M x 2 matrix containing the corresponding indices of
% assigned tracks and detections in its two columns. It also returns the
% indices of tracks and detections that remained unassigned. 

    function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()
        
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        
        % compute the cost of assigning each detection to each track
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);%损失矩阵计算
        end
        
        % solve the assignment problem
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);%匈牙利算法匹配
    end

%% Update Assigned Tracks
% The |updateAssignedTracks| function updates each assigned track with the
% corresponding detection. It calls the |correct| method of
% |vision.KalmanFilter| to correct the location estimate. Next, it stores
% the new bounding box, and increases the age of the track and the total
% visible count by 1. Finally, the function sets the invisible count to 0. 

    function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            
            % correct the estimate of the object's location
            % using the new detection
            correct(tracks(trackIdx).kalmanFilter, centroid);
            
            % replace predicted bounding box with detected
            % bounding box
            tracks(trackIdx).bbox = bbox;
            
            % update track's age
            tracks(trackIdx).age = tracks(trackIdx).age + 1;
            
            % update visibility
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
    end

%% Update Unassigned Tracks
% Mark each unassigned track as invisible, and increase its age by 1.

    function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
    end

%% Delete Lost Tracks
% The |deleteLostTracks| function deletes tracks that have been invisible
% for too many consecutive frames. It also deletes recently created tracks
% that have been invisible for too many frames overall. 

    function deleteLostTracks()
        if isempty(tracks)
            return;
        end
        
        invisibleForTooLong = 10;
        ageThreshold = 8;
        
        % compute the fraction of the track's age for which it was visible
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        
        % find the indices of 'lost' tracks
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
        
        % delete lost tracks
        tracks = tracks(~lostInds);
    end

%% Create New Tracks
% Create new tracks from unassigned detections. Assume that any unassigned
% detection is a start of a new track. In practice, you can use other cues
% to eliminate noisy detections, such as size, location, or appearance.

    function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        
        for i = 1:size(centroids, 1)
            
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            
            % create a Kalman filter object
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);
            
            % create a new track
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);
            
            % add it to the array of tracks
            tracks(end + 1) = newTrack;
            
            % increment the next id
            nextId = nextId + 1;
        end
    end

%% Display Tracking Results
% The |displayTrackingResults| function draws a bounding box and label ID 
% for each track on the video frame and the foreground mask. It then 
% displays the frame and the mask in their respective video players. 

    function displayTrackingResults()
        % convert the frame and the mask to uint8 RGB 
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        
        minVisibleCount = 8;
        if ~isempty(tracks)
              
            % noisy detections tend to result in short-lived tracks
            % only display tracks that have been visible for more than 
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            
            % display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % get bounding boxes
                bboxes = cat(1, reliableTracks.bbox);
                
                % get ids
                ids = int32([reliableTracks(:).id]);
                
                % create labels for objects indicating the ones for 
                % which we display the predicted rather than the actual 
                % location
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                
                % draw on the frame
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);
                
                % draw on the mask
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end
        
        % display the mask and the frame
        obj.maskPlayer.step(mask);        
        obj.videoPlayer.step(frame);
    end

%% Summary
% This example created a motion-based system for detecting and
% tracking multiple moving objects. Try using a different video to see if
% you are able to detect and track objects. Try modifying the parameters
% for the detection, assignment, and deletion steps.  
%
% The tracking in this example was solely based on motion with the
% assumption that all objects move in a straight line with constant speed.
% When the motion of an object significantly deviates from this model, the
% example may produce tracking errors. Notice the mistake in tracking the
% person labeled #12, when he is occluded by the tree. 
%
% The likelihood of tracking errors can be reduced by using a more complex
% motion model, such as constant acceleration, or by using multiple Kalman
% filters for every object. Also, you can incorporate other cues for
% associating detections over time, such as size, shape, and color. 

displayEndOfDemoMessage(mfilename)
end
