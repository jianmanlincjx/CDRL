exp_name=${1:-baseline}
eval_imgs=${2:-faces_aligned}
tmp_dir=exp/cmp_crossID/

for emo in neutral angry disgusted fear happy sad surprised
do
    printf "%-10s" "${emo}"
    python metrics/fid.py ${tmp_dir}/real/${emo} ${tmp_dir}/gen/${emo} --model resnet_face18
    python metrics/lse_cross_id.py ${tmp_dir}/gen/${emo} --data_dir /data2/JM/code/NED-main_CDRL/exp/txt --actors M003 M009 W029 M012 M030 W015 --skip_frame 2
    python metrics/csim.py ${tmp_dir}/real/${emo} ${tmp_dir}/gen/${emo} --actors M003 M009 W029 M012 M030 W015 
    echo ''
done

printf "======================================================="
printf ""

tmp_dir=exp/cmp_inter/
for emo in neutral angry disgusted fear happy sad surprised
do
    printf "%-10s" "${emo}"
    python metrics/fid.py ${tmp_dir}/real/${emo} ${tmp_dir}/gen/${emo} --model resnet_face18
    python metrics/lse_inter_id.py ${tmp_dir}/gen/${emo} --data_dir /data2/JM/code/NED-main_CDRL/exp/txt --actors M003 M009 W029 M012 M030 W015 --skip_frame 2
    python metrics/csim.py ${tmp_dir}/real/${emo} ${tmp_dir}/gen/${emo} --actors M003 M009 W029 M012 M030 W015 
    echo ''
done


##############################################################3
# ***1 # content + emotion
# interid
# neutral   FID: 0.722 LSE-D: 9.255 CSIM: 0.916 
# angry     FID: 1.045 LSE-D: 9.682 CSIM: 0.896 
# disgusted FID: 1.115 LSE-D: 9.213 CSIM: 0.927 
# fear      FID: 1.228 LSE-D: 9.490 CSIM: 0.908 
# happy     FID: 0.987 LSE-D: 9.427 CSIM: 0.923 
# sad       FID: 1.258 LSE-D: 9.243 CSIM: 0.911 
# surprised FID: 0.825 LSE-D: 9.327 CSIM: 0.918 
#           AVG: 1.025        9.377       0.914

# cross-id
# neutral   FID: 1.995 LSE-D: 9.393 CSIM: 0.849 
# angry     FID: 4.988 LSE-D: 9.307 CSIM: 0.740 
# disgusted FID: 4.687 LSE-D: 9.292 CSIM: 0.814 
# fear      FID: 5.021 LSE-D: 9.456 CSIM: 0.767 
# happy     FID: 3.264 LSE-D: 9.297 CSIM: 0.870 
# sad       FID: 5.479 LSE-D: 9.353 CSIM: 0.712 
# surprised FID: 4.976 LSE-D: 9.362 CSIM: 0.793 
#           AVG: 4.344        9.351       0.792
##############################################################3


##############################################################3
# ***1 # emotion
# interid
# neutral   FID: 0.764 LSE-D: 9.301 CSIM: 0.909 
# angry     FID: 1.093 LSE-D: 9.686 CSIM: 0.893 
# disgusted FID: 1.243 LSE-D: 9.331 CSIM: 0.920 
# fear      FID: 1.381 LSE-D: 9.497 CSIM: 0.908 
# happy     FID: 1.070 LSE-D: 9.383 CSIM: 0.922 
# sad       FID: 1.565 LSE-D: 9.380 CSIM: 0.903 
# surprised FID: 1.009 LSE-D: 9.217 CSIM: 0.919 
#           AVG: 1.161        9.399       0.911

# cross-id
# neutral   FID: 2.074 LSE-D: 9.454 CSIM: 0.851 
# angry     FID: 4.882 LSE-D: 9.385 CSIM: 0.708 
# disgusted FID: 4.927 LSE-D: 9.363 CSIM: 0.788 
# fear      FID: 5.136 LSE-D: 9.464 CSIM: 0.755 
# happy     FID: 3.340 LSE-D: 9.499 CSIM: 0.865 
# sad       FID: 5.534 LSE-D: 9.393 CSIM: 0.685 
# surprised FID: 4.920 LSE-D: 9.467 CSIM: 0.779 
#           AVG: 4.401        9.432       0.776
##############################################################3

##############################################################3
# ***1 # content
# inter-id
# neutral   FID: 0.673 LSE-D: 9.264 CSIM: 0.894
# angry     FID: 1.064 LSE-D: 9.636 CSIM: 0.896 
# disgusted FID: 1.184 LSE-D: 9.296 CSIM: 0.912 
# fear      FID: 1.447 LSE-D: 9.355 CSIM: 0.898 
# happy     FID: 0.919 LSE-D: 9.296 CSIM: 0.901
# sad       FID: 0.929 LSE-D: 9.417 CSIM: 0.886 
# surprised FID: 0.957 LSE-D: 9.427 CSIM: 0.885
#                1.024        9.384       0.896

# cross-id
# neutral   FID: 1.942 LSE-D: 9.387 CSIM: 0.827 
# angry     FID: 5.471 LSE-D: 9.380 CSIM: 0.731 
# disgusted FID: 4.583 LSE-D: 9.465 CSIM: 0.786 
# fear      FID: 5.035 LSE-D: 9.388 CSIM: 0.757 
# happy     FID: 3.701 LSE-D: 9.422 CSIM: 0.838 
# sad       FID: 5.009 LSE-D: 9.369 CSIM: 0.694 
# surprised FID: 4.858 LSE-D: 9.341 CSIM: 0.779 
#                4.374        9.393       0.774
##############################################################3


# \begin{table}[htbp]
# \centering
# \begin{tabular}{c|c|ccc}
# \toprule
# Settings & Methods & FAD$\downarrow$ & LSE-D$\downarrow$ & CSIM$\uparrow$ \\
# \hline
# \multirow{4}{*}{inter-id} & NED & 2.108 & 9.454 & 0.831 \\
# & Ours w/o CCRML  & 1.161 & 9.448 & 0.911 \\
# & Ours w/o audio  & --- & --- & --- \\
# & Ours CCRML-flex  & 1.287 & 9.399 & 0.901 \\
# & Ours & 1.025 & 9.378 & 0.914 \\
# \hline
# \multirow{4}{*}{cross-id} & NED & 4.448 & 9.906 & 0.773 \\
# & Ours w/o CCRML & 4.401 & 9.883 & 0.786 \\
# & Ours w/o audio  & --- & --- & --- \\
# & Ours CCRML-flex & 4.393 & 9.382 & 0.791 \\
# & Ours & 4.344 & 9.351 & 0.792 \\
# \bottomrule
# \end{tabular}
# \caption{The quantitative ablation analysis of CCRML.}
# \label{table: CCRML}
# \end{table}