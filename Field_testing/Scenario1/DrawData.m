close all
clear all
clc

load control.txt;
load automode.txt;
load traffic.txt;
load surrounding_obj.txt;
load decision.txt;

tmin = surrounding_obj(1200,1);
tmax = surrounding_obj(2000,1);
%%

control = control(control(:,1)>=tmin,:);
control = control(control(:,1)<=tmax,:);
figure
subplot(3,1,1)
plot(control(:,1)-tmin,control(:,2)*3.6);
xlabel('t/s')
ylabel('target speed(km/h)')
xlim([0, tmax-tmin])

for i = 1:1:length(control(:,3))
    while control(i,3) > 520
        control(i,3) = control(i,3) - 65536;
    end
    
    while control(i,3) < -520
        control(i,3) = control(i,3) + 65536;
    end
end

subplot(3,1,2)
plot(control(:,1)-tmin,control(:,3));
xlabel('t/s')
ylabel('steering angle/бу')
xlim([0, tmax-tmin])

automode = automode(automode(:,1)>=tmin,:);
automode = automode(automode(:,1)<=tmax,:);


subplot(3,1,3)
plot(automode(:,1)-tmin,automode(:,2)*0.5,'.');
ylim([0 1.5])
xlabel('t/s')
ylabel('autonomous driving mode')
xlim([0, tmax-tmin])


d = 0;
ego_x_last = traffic(1,4);
ego_y_last = traffic(1,5);
for i = 2:1:length(traffic(:,1))
    ego_x = traffic(i,4);
    ego_y = traffic(i,5);
    d = d + sqrt((ego_x-ego_x_last)^2+(ego_y_last-ego_y)^2);
    ego_x_last = traffic(i,4);
    ego_y_last = traffic(i,5);
end


%%
surrounding_t = surrounding_obj(surrounding_obj(:,1)>=tmin,:);
surrounding_vehicle = surrounding_t(surrounding_t(:,1)<=tmax,:);

ego_pose_t = traffic(traffic(:,1)>=tmin,:);
ego_pose = ego_pose_t(ego_pose_t(:,1)<=tmax,:);

% surrounding_vehicle = surrounding_obj(surrounding_obj(:,1)==t,:);

figure
plot(surrounding_vehicle(:,2),surrounding_vehicle(:,3),'o')
hold on 
plot(ego_pose(:,4),ego_pose(:,5),'o')
legend('surrounding obj pose','ego pose')
xlabel('UTM x/m')
ylabel('UTM y/m')


%% Driving Data Process



visited_times_rule = zeros(length(automode(:,1)),1);

visited_times_RL = zeros(length(automode(:,1)),1);

c_rule = ones(length(automode(:,1)),1);
c_RL = -ones(length(automode(:,1)),1);

figure
subplot(2,1,1)
plot(automode(:,1)-tmin, visited_times_rule);
xlim([0 tmax-tmin])
xlabel('t/s')
ylabel('Case Num in Dataset (ln)')

subplot(2,1,2)
plot(automode(:,1)-tmin, smoothdata(c_rule,'lowess'));
xlim([0 tmax-tmin])
ylim([-1.5 1.5])
xlabel('t/s')
ylabel('Confidence Value')

hold on
plot(automode(:,1)-tmin, smoothdata(c_RL,'lowess'));
legend('original policy','other policy')


