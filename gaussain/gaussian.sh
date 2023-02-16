source /etc/profile
# export HADOOP_USER_NAME=ci_rcmd;
# export HADOOP_USER_RPCPASSWORD=LtZ1DQpzrfAt;

# INPUT="hdfs://R2/projects/ci_rcmd/hdfs/prod/recommend/video_feature/deep_model/pb/v_6/{2022-02-20}/00/*"
INPUT="hdfs://R2/projects/ci_rcmd/hdfs/prod/recommend/video_feature/deep_model/pb/v_6/{2022-04-02}/*"
OUTPUT="hdfs://R2/projects/ci_rcmd/hdfs/dev/video/sean.sun/pso/raw_data"

echo ${INPUT}
/usr/share/spark-2.4/bin/spark-submit \
        --master yarn \
        --deploy-mode client \
        --queue=video-rec \
        --num-executors 600 \
        --executor-memory 6g \
        --driver-memory 20g \
        --executor-cores 4 \
        --jars /ldap_home/sean.sun/gaussian_process/spark-tfrecord_2.11-0.3.4_0107.jar \
        --py-files /ldap_home/sean.sun/gaussian_process/video_proto.zip,/ldap_home/sean.sun/gaussian_process/video_util.zip,/ldap_home/sean.sun/multi_score_combine/gaussian/gaussian_model_utils.py \
        grid_search_distribution.py \
        --input ${INPUT} \
        --output ${OUTPUT} \
