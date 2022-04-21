function S = measure_texture_stats_copy(sample_sound, P, measurement_win, audio_filts, mod_filts)

% function S = MEASURE_TEXTURE_STATS(SAMPLE_SOUND, P, MEASUREMENT_WIN, ...
%   AUDIO_FILTS, AUDIO_CUTOFFS_HZ, MOD_FILTS, ENV_AC_FILTS, MOD_C1_FILTS, ...
%   MOD_C2_FILTS, SUB_HIST_BINS, SUB_ENV_HIST_BINS)
%
% Function to measure texture statistics from a sound signal.
%
% MEASUREMENT_WIN specifies the weights on each sample of 
% the signal in the average
%
% subsequent arguments (filters etc.) are optional and are created by
% function if not provided or if provided as empty ([]) arguments
%
% other parameters are specified in P (and can be set using
% SYNTHESIS_PARAMETERS.M)
%

% This code is part of an instantiation of a sound texture synthesis
% algorithm developed with Eero Simoncelli and described in this paper:
%
% McDermott, J.H., Simoncelli, E.P. (2011) Sound texture perception via
% statistics of the auditory periphery: Evidence from sound synthesis.
% Neuron, 71, 926-940. 
%
% Dec 2012 -- Josh McDermott <jhm@mit.edu>
% May 2014 -- modified to fix incompatibility with overcomplete filter
% banks -- Josh McDermott

%generate subbands and subband envelopes from which statistics are measured
subbands = generate_subbands(sample_sound, audio_filts);
%analytic_subbands = hilbert(subbands);
subband_envs = abs(hilbert(subbands));
if P.compression_option==1 %power compression
    subband_envs = subband_envs.^P.comp_exponent;
elseif P.compression_option==2 %log compression
    subband_envs = log10(subband_envs+P.log_constant);
end
ds_factor=P.audio_sr/P.env_sr;
subband_envs = resample(subband_envs,1,ds_factor);
subband_envs(subband_envs<0)=0;

for j=1:size(audio_filts,2) %go through subbands
    S.subband_skew(j) = skewness(subbands(:,j));
    S.subband_kurt(j) = kurtosis(subbands(:,j));
    S.env_mean(j) = stat_central_moment_win(subband_envs(:,j),1,measurement_win);
    S.env_var(j) = stat_central_moment_win(subband_envs(:,j),2,measurement_win,S.env_mean(j));
    S.env_skew(j) = stat_central_moment_win(subband_envs(:,j),3,measurement_win,S.env_mean(j));
    S.env_kurt(j) = stat_central_moment_win(subband_envs(:,j),4,measurement_win,S.env_mean(j));

    S.mod_power(j,1:P.N_mod_channels) = stat_mod_power_win(subband_envs(:,j), mod_filts, P.use_zp, measurement_win);

end
