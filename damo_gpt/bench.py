from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

if __name__ == '__main__':
    input = '程序员脱发用什么洗发水'
    model_id = 'damo/nlp_gpt3_text-generation_2.7B'
    pipe = pipeline(Tasks.text_generation, model=model_id)

    # 可以在 pipe 中输入 max_length, top_k, top_p, temperature 等生成参数
    print(pipe(input, max_length=512))
