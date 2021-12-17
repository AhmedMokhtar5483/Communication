%% Generate the data stream:
clear
close all

N = 1200;
data = randi([0,1],N,4);
%% Bulit 16-QAM:
QAM_data = data(:,1) + data(:,2)*2 + data(:,3)*4+ data(:,4)*8;

% 1) Mapper:
%-----------
QAM_data_mapped_all = (QAM_data == 0)*(-3-3i) + (QAM_data == 1)*(-3-1i)...
    +(QAM_data == 2 )*(-3+3i) + (QAM_data == 3)*(-3+1i)...
    +(QAM_data == 4 )*(-1-3i) + (QAM_data == 5)*(-1-1i)...
    +(QAM_data == 6 )*(-1+3i) + (QAM_data == 7)*(-1+1i)...
    +(QAM_data == 8 )*(3-3i) + (QAM_data == 9)*(3-1i)...
    +(QAM_data == 10)*(3+3i) + (QAM_data == 11)*(3+1i)...
    +(QAM_data == 12)*(1-3i) + (QAM_data == 13)*(1-1i)...
    +(QAM_data == 14)*(1+3i) + (QAM_data == 15)*(1+1i);

QAM_data_mapped = [real(QAM_data_mapped_all) , imag(QAM_data_mapped_all)];

% 2) The channel:
%----------------
v = randn(N,2);
R = sqrt(sum(v.^2,2))/sqrt(2);
QAM_data_channel = R.*QAM_data_mapped + v;
% QAM_data_channel = QAM_data_mapped;
% 3) Reciver:
%------------
QAM_data_recived = zeros(N,4); % Intialize Matrix After the demodulator

QAM_data_recived(:,1) = ((QAM_data_channel(:,2)>-2) & (QAM_data_channel(:,2)<2))*1;
QAM_data_recived(:,2) = (QAM_data_channel(:,2)>0)*1;
QAM_data_recived(:,3) = ((QAM_data_channel(:,1)>-2) & (QAM_data_channel(:,1)<2))*1;
QAM_data_recived(:,4) = (QAM_data_channel(:,1)>0)*1;
% QAM_data_recived = QAM_data_recived./R;
% show the output figure for each stage:
%---------------------------------------
figure
subplot(1,2,1)
scatter(QAM_data_mapped(:,1), QAM_data_mapped(:,2),50,'*')
Ax = gca; % set origin (0,0)
Ax.XAxisLocation = 'origin';
Ax.YAxisLocation = 'origin';
axis([-4  4 -4  4])
title('16-QAM constelation system after mapping')
xlabel('In-Phase componant')
ylabel('Quadrature componant')
grid

subplot(1,2,2)
scatter(QAM_data_channel(:,1), QAM_data_channel(:,2),50,'*')
Ax = gca; % set origin (0,0)
Ax.XAxisLocation = 'origin';
Ax.YAxisLocation = 'origin';
axis([-4  4  -4  4])
title('16-QAM constelation system after Channel')
xlabel('In-Phase componant')
ylabel('Quadrature componant')
grid

N0 = 0.01:0.01:10;
N0_len = length(N0);
BER_QAM = zeros(1,N0_len);

for i = 1:N0_len
    % 2) The channel:
    %----------------
    v = sqrt(N0(i)/2)*randn(N,2);
%     R = sqrt(sum(v.^2,2))/(sqrt(2)*sqrt(N0(i)/2));
    QAM_data_channel = R.*QAM_data_mapped + v;
    QAM_data_channel = QAM_data_channel./R;
%     QAM_data_channel = QAM_data_mapped;
    % 3) Reciver:
    %------------   3421
    QAM_data_recived(:,1) = ((QAM_data_channel(:,2)>-2) & (QAM_data_channel(:,2)<2))*1;
    QAM_data_recived(:,2) = (QAM_data_channel(:,2)>0)*1;
    QAM_data_recived(:,3) = ((QAM_data_channel(:,1)>-2) & (QAM_data_channel(:,1)<2))*1;
    QAM_data_recived(:,4) = (QAM_data_channel(:,1)>0)*1;
    
    % 4) BER Calculator:
    %-------------------
    BER_QAM(i) = (4*N - sum(sum(data==QAM_data_recived)))/(4*N);
end

%% Bulit Coded encoder 16-QAM (1/3):
n=3;
M = n*N;
BER_coded_QAM = zeros(1,N0_len);
% 0) Encoder:
%------------
data_coded = repelem(data,3,1);

% 1) Mapper:
%-----------
QAM_data_coded = data_coded(:,1) + data_coded(:,2)*2 + data_coded(:,3)*4+ data_coded(:,4)*8;

QAM_data_mapped_all_coded = (QAM_data_coded == 0)*(-3-3i) + (QAM_data_coded == 1)*(-3-1i)...
    +(QAM_data_coded == 2 )*(-3+3i) + (QAM_data_coded == 3)*(-3+1i)...
    +(QAM_data_coded == 4 )*(-1-3i) + (QAM_data_coded == 5)*(-1-1i)...
    +(QAM_data_coded == 6 )*(-1+3i) + (QAM_data_coded == 7)*(-1+1i)...
    +(QAM_data_coded == 8 )*(3-3i) + (QAM_data_coded == 9)*(3-1i)...
    +(QAM_data_coded == 10)*(3+3i) + (QAM_data_coded == 11)*(3+1i)...
    +(QAM_data_coded == 12)*(1-3i) + (QAM_data_coded == 13)*(1-1i)...
    +(QAM_data_coded == 14)*(1+3i) + (QAM_data_coded == 15)*(1+1i);

QAM_data_mapped_coded = [real(QAM_data_mapped_all_coded)/sqrt(3) , imag(QAM_data_mapped_all_coded)/sqrt(3)];
QAM_data_recived_decoded = zeros(N,4);
v = randn(M,2);
R = sqrt(sum(v.^2,2))/(sqrt(2));

for i = 1:N0_len
    % 2) The channel:
    %----------------
    v = sqrt(N0(i)/2)*randn(M,2);
%     R = sqrt(sum(v.^2,2))/(sqrt(2)*sqrt(N0(i)/2));
    QAM_data_channel_coded = R.*QAM_data_mapped_coded + v;
    QAM_data_channel_coded = QAM_data_channel_coded./R;
%     QAM_data_channel_coded = QAM_data_mapped_coded;
    % 3) Reciver:
    %------------   
    QAM_data_recived_coded(:,1) = ((QAM_data_channel_coded(:,2)>-2/sqrt(3)) & (QAM_data_channel_coded(:,2)<2/sqrt(3)))*1;
    QAM_data_recived_coded(:,2) = (QAM_data_channel_coded(:,2)>0)*1;
    QAM_data_recived_coded(:,3) = ((QAM_data_channel_coded(:,1)>-2/sqrt(3)) & (QAM_data_channel_coded(:,1)<2/sqrt(3)))*1;
    QAM_data_recived_coded(:,4) = (QAM_data_channel_coded(:,1)>0)*1;
    
    % 4) Decoder:
    %------------
    for j = 1:4
        for k =1:3:M
            QAM_data_recived_decoded(floor(k/3)+1,j) = (sum(QAM_data_recived_coded(k:k+2,j))>1)*1;
        end
    end
    % 5) BER Calculator:
    %-------------------
    BER_coded_QAM(i) = (4*N - sum(sum(data==QAM_data_recived_decoded)))/(4*N);
end

figure
semilogy(10*log(2.5./(N0)),BER_coded_QAM)
hold on
semilogy(10*log(2.5./(N0)),BER_QAM)
legend('Coded 16-QAM BER','UnCoded 16-QAM BER')
grid
%% Clean variables to reduce memorey usage:
clear i j k b Ax v R

clear QAM_data_channel QAM_data_channel_coded QAM_data_decoded QAM_data_demaped
clear QAM_data_recived_decoded QAM_data_mapped QAM_data_mapped_coded QAM_data_recived
clear QAM_data_recived_coded QAM_data_mapped_all_coded QAM_data_mapped_all
clear QAM_data_coded QAM_data

%% Bulit QPSK:

QPSK_data = data(:,1) + data(:,2)*2;
QPSK_data_mapped_all = (QPSK_data == 0)*(-1-1i) + (QPSK_data == 1)*(-1+1i)...
    +(QPSK_data == 2)*(1-1i) + (QPSK_data == 3)*(1+1i);

QPSK_data_mapped = [real(QPSK_data_mapped_all) , imag(QPSK_data_mapped_all)];

% 2) The channel:
%----------------
v = randn(N,2);
R = sqrt(sum(v.^2,2));
QPSK_data_channel = R.*QPSK_data_mapped + v;
% QPSK_data_channel = QPSK_data_mapped;

% 3) Reciver:
%------------
QPSK_data_recived = zeros(N,2);
QPSK_data_recived(:,1) = (QPSK_data_channel(:,2)>0);
QPSK_data_recived(:,2) = (QPSK_data_channel(:,1)>0);

% show the output figure for each stage:
%---------------------------------------
figure
subplot(1,2,1)
scatter(QPSK_data_mapped(:,1), QPSK_data_mapped(:,2),100,'*')
Ax = gca; % set origin (0,0)
Ax.XAxisLocation = 'origin';
Ax.YAxisLocation = 'origin';
axis([-4  4    -4  4])
title('QPSK constelation system')
xlabel('In-Phase componant')
ylabel('Quadrature componant')
grid

subplot(1,2,2)
scatter(QPSK_data_channel(:,1), QPSK_data_channel(:,2),50,'*')
Ax = gca; % set origin (0,0)
Ax.XAxisLocation = 'origin';
Ax.YAxisLocation = 'origin';
axis([-4  4    -4  4])
title('16-QAM constelation system after Channel')
xlabel('In-Phase componant')
ylabel('Quadrature componant')
grid



BER_QPSK = zeros(1,N0_len);

for i = 1:N0_len
    % 2) The channel:
    %----------------
    v = sqrt(N0(i)/2)*randn(N,2);
%     R = sqrt(sum(v.^2,2))/(sqrt(2)*sqrt(N0(i)/2));
    QPSK_data_channel = R.*QPSK_data_mapped + v;
    QPSK_data_channel = QPSK_data_channel ./R; 
    % 3) Reciver:
    %------------
    QPSK_data_recived(:,1) = (QPSK_data_channel(:,2)>0);
    QPSK_data_recived(:,2) = (QPSK_data_channel(:,1)>0);
    
    % 4) BER Calculator:
    %-------------------
    BER_QPSK(i) = (2*N - sum(sum(data(:,1:2)==QPSK_data_recived)))/(2*N);
end

%% bulit Coded encoder QPSK (1/3):

QPSK_data_coded = data_coded(:,1) + data_coded(:,2)*2;
QPSK_data_mapped_all_coded = (QPSK_data_coded == 0)*(-1-1i) + (QPSK_data_coded == 1)*(-1+1i)...
    +(QPSK_data_coded == 2)*(1-1i) + (QPSK_data_coded == 3)*(1+1i);

QPSK_data_mapped_coded = [real(QPSK_data_mapped_all_coded)/sqrt(3) , imag(QPSK_data_mapped_all_coded)/sqrt(3)];

QPSK_data_recived_coded = zeros(M,2);
QPSK_data_recived_decoded = zeros(N,2);
BER_coded_QPSK = zeros(1,N0_len);

v = randn(M,2);
R = sqrt(sum(v.^2,2))/sqrt(2);

for i = 1:N0_len
    % 2) The channel:
    %----------------
    v = sqrt(N0(i)/2)*randn(M,2);
%     R = sqrt(sum(v.^2,2))/(sqrt(2)*sqrt(N0(i)/2));
    QPSK_data_channel_coded = R.*QPSK_data_mapped_coded + v;
    QPSK_data_channel_coded = QPSK_data_channel_coded./R;
    % 3) Reciver:
    %------------
    QPSK_data_recived_coded(:,1) = (QPSK_data_channel_coded(:,2)>0);
    QPSK_data_recived_coded(:,2) = (QPSK_data_channel_coded(:,1)>0);
    
    % 4) Decoder:
    %------------
    for j = 1:2
        for k =1:3:M
            QPSK_data_recived_decoded(floor(k/3)+1,j) = (sum(QPSK_data_recived_coded(k:k+2,j))>1)*1;
        end
    end
    
    % 5) BER Calculator:
    %-------------------
    BER_coded_QPSK(i) = (2*N - sum(sum(data(:,1:2)==QPSK_data_recived_decoded)))/(2*N);
end

% Show output figure of coded encoder and uncoded encoder QAM:
%-------------------------------------------------------------
figure
semilogy(10*log(1./N0),BER_coded_QPSK)
hold on
semilogy(10*log(1./(N0)),BER_QPSK)
legend('Coded QPSK BER','UnCoded QPSK BER')
grid
%% Clean variables to reduce memorey usage:
clear i j k b Ax v R

clear QPSK_data_channel QPSK_data_channel_coded QPSK_data_decoded QPSK_data_demaped
clear QPSK_data_recived_decoded QPSK_data_mapped QPSK_data_mapped_coded QPSK_data_recived
clear QPSK_data_recived_coded QPSK_data_mapped_all_coded QPSK_data_mapped_all
clear QPSK_data_coded QPSK_data

clear BER_QPSK BER_QAM BER_coded_QAM BER_coded_QPSK

clear data data_coded N M n N0 N0_len
%% Generate data for OFDM system simulation Part
N = 252;
n = 3;
M = n*N;
data = randi([0,1],N,1);
data_coded = repelem(data,n);
%% Bulit 16-QAM:

% 1) Interliver:
%---------------
Ncbps_QAM =  256; %
QAM_seq = zeros(1,Ncbps_QAM);
for k = 0:Ncbps_QAM-1
    QAM_seq(k+1) = (Ncbps_QAM*mod(k,16)/16)+ floor(k/16)+1;
end

x = 4*63;  % 4: number of bits ber symbols, 63: number of symbols
if (mod(N,x)~=0)
    QAM_interliving = [data; zeros(x-mod(N,x),1)];
else
    QAM_interliving = data;
end

QAM_interliving = reshape(QAM_interliving,[x,ceil(length(QAM_interliving)/x)]);
QAM_interliving = [QAM_interliving; zeros(4,size(QAM_interliving,2))];

QAM_interliving = intrlv(QAM_interliving,int16(QAM_seq));
QAM_interlived = reshape(QAM_interliving,64,[],4);

% 2) Mapper:
%-----------
QAM_data = QAM_interlived(:,:,1) + QAM_interlived(:,:,2)*2 ...
    + QAM_interlived(:,:,3)*4+ QAM_interlived(:,:,4)*8;

QAM_data_mapped_all = (QAM_data == 0)*(-3-3i) + (QAM_data == 1)*(-3-1i)...
    +(QAM_data == 2 )*(-3+3i) + (QAM_data == 3)*(-3+1i)...
    +(QAM_data == 4 )*(-1-3i) + (QAM_data == 5)*(-1-1i)...
    +(QAM_data == 6 )*(-1+3i) + (QAM_data == 7)*(-1+1i)...
    +(QAM_data == 8 )*(3-3i) + (QAM_data == 9)*(3-1i)...
    +(QAM_data == 10)*(3+3i) + (QAM_data == 11)*(3+1i)...
    +(QAM_data == 12)*(1-3i) + (QAM_data == 13)*(1-1i)...
    +(QAM_data == 14)*(1+3i) + (QAM_data == 15)*(1+1i);

% 3) IFFT:
%---------
QAM_IFFT = ifft(QAM_data_mapped_all,64);

%% Bulit coded 16-QAM:

% 1) Interliver:
%---------------
Ncbps_QAM =  256; %
QAM_seq = zeros(1,Ncbps_QAM);
for k = 0:Ncbps_QAM-1
    QAM_seq(k+1) = (Ncbps_QAM*mod(k,16)/16)+ floor(k/16)+1;
end

x = 4*3*21;  % 4: number of bits ber symbols, 21: number of symbols, 3: number of Coding
if (mod(M,x)~=0)
    QAM_interliving_coded = [data_coded; zeros(x-mod(M,x),1)];
else
    QAM_interliving_coded = data_coded;
end
QAM_interliving_coded = reshape(QAM_interliving_coded,[x,length(QAM_interliving_coded)/x]);
QAM_interliving_coded = [QAM_interliving_coded; zeros(4,size(QAM_interliving_coded,2))];

QAM_interliving_coded = intrlv(QAM_interliving_coded,int16(QAM_seq));
QAM_interlived_coded = reshape(QAM_interliving_coded,64,[],4);

% 2) Mapper:
%-----------
QAM_data_coded = QAM_interlived_coded(:,:,1) + QAM_interlived_coded(:,:,2)*2 + QAM_interlived_coded(:,:,3)*4+ QAM_interlived_coded(:,:,4)*8;

QAM_data_mapped_all_coded = (QAM_data_coded == 0)*(-3-3i) + (QAM_data_coded == 1)*(-3-1i)...
    +(QAM_data_coded == 2 )*(-3+3i) + (QAM_data_coded == 3)*(-3+1i)...
    +(QAM_data_coded == 4 )*(-1-3i) + (QAM_data_coded == 5)*(-1-1i)...
    +(QAM_data_coded == 6 )*(-1+3i) + (QAM_data_coded == 7)*(-1+1i)...
    +(QAM_data_coded == 8 )*(3-3i) + (QAM_data_coded == 9)*(3-1i)...
    +(QAM_data_coded == 10)*(3+3i) + (QAM_data_coded == 11)*(3+1i)...
    +(QAM_data_coded == 12)*(1-3i) + (QAM_data_coded == 13)*(1-1i)...
    +(QAM_data_coded == 14)*(1+3i) + (QAM_data_coded == 15)*(1+1i);

% 3) IFFT:
%---------
QAM_IFFT_coded = ifft(QAM_data_mapped_all_coded,64);

%% Bulit coded QPSK:

% 1) Interliver:
%---------------

Ncbps_QPSK =  128; 
x = 2*3*21;  % 2:# bits ber symbols, 21: # of symbols 3: #Coding
QPSK_seq = zeros(1,Ncbps_QPSK);
for k = 0:Ncbps_QPSK-1
    QPSK_seq(k+1) = (Ncbps_QPSK*mod(k,16)/16)+ floor(k/16)+1;
end

if (mod(M,x)~=0)
    QPSK_interlived = [data_coded; zeros(x-mod(M,x),4)];
else
    QPSK_interlived = data_coded;
end
% QPSK_interlived = reshape(QPSK_interlived,[x,length(QPSK_interlived)/x,4]);
% QPSK_interlived = [QPSK_interlived, zeros(2,size(QPSK_interlived,2))];
% QPSK_interlived = intrlv(QPSK_interlived,int16(QPSK_seq));

%%





















