# Extracted Notes

## 2109.07958 - TruthfulQA: Measuring How Models Mimic Human Falsehoods

### Skim (Chunk 1)
TruthfulQA: Measuring How Models Mimic Human Falsehoods Stephanie Lin University of Oxford sylin07@gmail.com Jacob Hilton OpenAI jhilton@openai.com Owain Evans University of Oxford owaine@gmail.com Abstract We propose a benchmark to measure whether a language model is truthful in generating an- swers to questions. The benchmark comprises 817 questions that span 38 categories, includ- ing health, law, ﬁnance and politics. We crafted questions that some humans would an- swer falsely due to a false belief or miscon- ception. To perform well, models must avoid generating false answers learned from imitat- ing human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a T5-based model. The best model was truthful on 58% of questions, while hu- man performance was 94%. Models generated many false answers that mimic popular mis- conceptions and have the potential to deceive humans. The largest models w

## 2110.14168 - Training Verifiers to Solve Math Word Problems

### Skim (Chunk 1)
Training Veriﬁers to Solve Math Word Problems Karl Cobbe ∗ Vineet Kosaraju ∗ Mohammad Bavarian Mark Chen Heewoo Jun Lukasz Kaiser Matthias Plappert Jerry Tworek Jacob Hilton Reiichiro Nakano Christopher Hesse John Schulman OpenAI Abstract State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathe- matical reasoning. To diagnose the failures of current models and support research, we introduce GSM8K, a dataset of 8.5K high quality linguisti- cally diverse grade school math word problems. We ﬁnd that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution. To increase per- formance, we propose training veriﬁers to judge the correctness of model completions. At test time, we generate many candidate solutions and select the one ranked

## 2303.11366 - Reflexion: Language Agents with Verbal Reinforcement Learning

### Skim (Chunk 1)
Reflexion: Language Agents with Verbal Reinforcement Learning Noah Shinn Northeastern University noahshinn024@gmail.com Federico Cassano Northeastern University cassano.f@northeastern.edu Edward Berman Northeastern University berman.ed@northeastern.edu Ashwin Gopinath Massachusetts Institute of Technology agopi@mit.edu Karthik Narasimhan Princeton University karthikn@princeton.edu Shunyu Yao Princeton University shunyuy@princeton.edu Abstract Large language models (LLMs) have been increasingly used to interact with exter- nal environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require exten- sive training samples and expensive model fine-tuning. We proposeReflexion, a novel framework to reinforce language agents not b

### Deep Read Notes (all chunks)
- 2303.11366_reflexion_chunk_001.pdf: Reflexion: Language Agents with Verbal Reinforcement Learning Noah Shinn Northeastern University noahshinn024@gmail.com Federico Cassano Northeastern University cassano.f@northeastern.edu Edward Berman Northeastern University berman.ed@northeastern.edu Ashwin 
  - datasets_mentioned: HumanEval
  - metrics_mentioned: accuracy, em, pass@1
- 2303.11366_reflexion_chunk_002.pdf: ActionObs / Reward Trajectory (short-term memory) Experience (long-term memory) Self-reflection (LM) Agent Actor (LM) Environment Evaluator (LM) External feedback Internal feedback Reflective text Algorithm 1 Reinforcement via self-reflection Initialize Actor,
  - datasets_mentioned: HotpotQA, HumanEval
  - metrics_mentioned: EM, accuracy, em, exact match
- 2303.11366_reflexion_chunk_003.pdf: 0 2 4 6 Trial Number 0.2 0.4 0.6 0.8Proportion of Solved Tasks (a) HotPotQA Success Rate CoT only ReAct only CoT + Reflexion ReAct + Reflexion 0 1 2 3 4 5 6 7 Trial Number 0.4 0.6 0.8 1.0Proportion of Solved Tasks (b) HotPotQA CoT (GT) CoT (GT) only CoT (GT) +
  - datasets_mentioned: HumanEval
  - metrics_mentioned: Pass@1, accuracy, em, pass@1
- 2303.11366_reflexion_chunk_004.pdf: References [1] Ahn, M., Brohan, A., Brown, N., Chebotar, Y ., Cortes, O., David, B., Finn, C., Gopalakrishnan, K., Hausman, K., Herzog, A., et al. (2022). Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691. [2
  - datasets_mentioned: HotpotQA, HumanEval
  - metrics_mentioned: EM, Em, Pass@1, accuracy, em
- 2303.11366_reflexion_chunk_005.pdf: B Decision-making Environment: You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a s
  - datasets_mentioned: HumanEval
  - metrics_mentioned: em
- 2303.11366_reflexion_chunk_006.pdf: (Function implementation) (Unit test feedback) (Self-reflection) (Instruction for next function implmentation) 16 D Reasoning D.1 Full example Trial #1: Question: Grown-Ups starred the actor who was best known for which role on "’Allo ’Allo!"? Thought 1: I nee
  - metrics_mentioned: em
- 2303.11366_reflexion_chunk_007.pdf: D.4 HotPotQA episodic memory (EPM) ablation prompts D.4.1 (EPM) Chain-of-Thought + Reflexion Trial #1: Question: Which of Jonny Craig and Pete Doherty has been a member of more bands ? Thought 1: Let’s think step by step. Jonny Craig has been a member of six b
  - metrics_mentioned: Em, em

## 2303.17651 - Self-Refine: Iterative Refinement with Self-Feedback

### Skim (Chunk 1)
SELF -R EFINE : Iterative Refinement with Self-Feedback Aman Madaan1, Niket Tandon2, Prakhar Gupta1, Skyler Hallinan3, Luyu Gao1, Sarah Wiegreffe2, Uri Alon1, Nouha Dziri2, Shrimai Prabhumoye4, Yiming Yang1, Shashank Gupta2, Bodhisattwa Prasad Majumder5, Katherine Hermann6, Sean Welleck2,3, Amir Yazdanbakhsh6, Peter Clark2 1Language Technologies Institute, Carnegie Mellon University 2Allen Institute for Artificial Intelligence 3University of Washington 4NVIDIA 5UC San Diego 6Google Research, Brain Team amadaan@cs.cmu.edu, nikett@allenai.org Abstract Like humans, large language models ( LLM s) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce SELF -REFINE , an approach for improving initial outputs from LLM s through iterative feedback and refinement. The main idea is to generate an initial output using an LLM ; then

### Deep Read Notes (all chunks)
- 2303.17651_self_refine_chunk_001.pdf: SELF -R EFINE : Iterative Refinement with Self-Feedback Aman Madaan1, Niket Tandon2, Prakhar Gupta1, Skyler Hallinan3, Luyu Gao1, Sarah Wiegreffe2, Uri Alon1, Nouha Dziri2, Shrimai Prabhumoye4, Yiming Yang1, Shashank Gupta2, Bodhisattwa Prasad Majumder5, Kathe
  - metrics_mentioned: eM, em
- 2303.17651_self_refine_chunk_002.pdf: Here, the prompt pfb provides examples of feedback in the form of input-output-feedback triples ⟨x(k), y(k), f b(k)⟩. We prompt the model to write feedback that is actionable and specific via f b(k). By ‘actionable’, we mean the feedback should contain a concr
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_003.pdf: Task y0 y1 y2 y3 Code Opt. 22.0 27.0 27.9 28.8 Sentiment Rev. 33.9 34.9 36.1 36.8 Constrained Gen. 29.0 40.3 46.7 49.7 ∆(y0→y1) ∆(y1→y2) ∆(y2→y3) 0 5 10 5 0.9 0.9 11.3 6.4 3 1 1.2 0.7 C. Opt. C. Gen. S. Rev. Figure 4: Left: Iteration-wise score improvements. E
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_004.pdf: hope that our iterative approach will help drive further research in this area. To this end, we make all our code, data and prompts anonymously available at https://selfrefine.info/. References Teresa M. Amabile. 1983. A Theoretical Framework. In The Social Ps
  - metrics_mentioned: EM, Em, em
- 2303.17651_self_refine_chunk_005.pdf: Niket Tandon, Aman Madaan, Peter Clark, and Yiming Yang. 2022. Learning to repair: Repairing model output errors after deployment using a dynamic memory of feedback. In Findings of the Association for Computational Linguistics: NAACL 2022, pages 339–352. Hugo 
  - datasets_mentioned: GSM8K
  - metrics_mentioned: Em, F1, accuracy, em
- 2303.17651_self_refine_chunk_006.pdf: C Human Evaluation The A/B evaluation in our study was conducted by the authors, where a human judge was presented with an input, task instruction, and two candidate outputs generated by the baseline method and SELF -REFINE . The setup was blind, i.e., the jud
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_007.pdf: Method %OPT) Puri et al. (2021) Human References 38.2 OpenAI Models: OpenAI (2022, 2023) CODEX 13.1 GPT-3.5 14.8 ChatGPT 22.2 GPT-4 27.3 Nijkamp et al. (2022) C ODE GEN-16B 1.1 Berger et al. (2022) SCALENE 1.4 SCALENE (BEST @16) 12.6 SCALENE (BEST @32) 19.6 Ma
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_008.pdf: 0 10 20 30 40 50 60 70 80 90 100 SELF-REFINE SELF-REFINE 27.2 15.5 35.6 51.1 37.2 33.3 Preference rates for Sentiment Reversal MULTI ChatGPT27.2 15.5 35.6 51.1 37.2 33.3 0 10 20 30 40 50 60 70 80 90 100 SELF-REFINE SELF-REFINE 11.4 6.1 45.4 53.82 43.2 40.05 Pr
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_009.pdf: I Beyond Benchmarks SELF -REFINE demonstrates its iterative feedback and refinement capabilities in the context of website layout generation. ChatGPT initially produces a rudimentary layout for a given topic, and then uses the FEEDBACK to suggest specific, act
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_010.pdf: J Statistical Confidence Intervals GPT-3.5 ChatGPT GPT-4 Task Base +S ELF -REFINE Base +S ELF -REFINE Base +S ELF -REFINE Sentiment Reversal 8.8 ± 2.05 30.4 ± 3.61∗ 11.4 ± 2.34 43.2 ± 3.98∗ 3.8 ± 1.28 36.2 ± 3.82∗ Dialogue Response 36.4 ± 6.14 63.6 ± 6.62∗ 40.
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_011.pdf: Starting Code: v0 print((int((int(eval(input()))+1)/2))) Code v1 print( (int( (int(eval(input())) + 1) / 2 ) ) Code v2 num_input = eval(input()) num_input = int(num_input) num_input += 1 num_result = int(num_input / 2) print(num_result) Figure 12: SELF -REFINE
  - metrics_mentioned: em, win rate
- 2303.17651_self_refine_chunk_012.pdf: O Math Reasoning We use the Grade School Math 8k (GSM-8k) dataset (Cobbe et al., 2021) for evaluatingSELF -REFINE on math reasoning. In the context of grade school mathematics, SELF -R EFINE aims to enable LLMs to iteratively refine their mathematical problem-
  - metrics_mentioned: Accuracy, accuracy, em
- 2303.17651_self_refine_chunk_013.pdf: Criteria output from GPT3:STSLWN output from SELF-REFINE: Seq2Seq Ease of pronunciation Pronounced as ess-tee-ess-ell-double- you-enn which is very difficult. Pronounced as seq-two-seq which is easy. Ease of spelling Very difficult to spell. Easy to spell. Rel
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_014.pdf: Title: Underwater Breathing Product with no Accessories Acronym: UBPA Scores: * Ease of pronunciation: UBPA is pronounced "uhb-puh". This is an easy acronym to pronounce. 4/5 * Ease of spelling: UBPA is easy to spell. 4/5 * Relation to title: UBPA stands for "
- 2303.17651_self_refine_chunk_015.pdf: a, b = input().split() n = int(a + b) flag = False for i in range(n): if i ** 2 == n: flag = True break print('Yes' if flag else 'No') # Why is this code slow? # This code is slow because it is using a brute force approach to find the square root of the input 
- 2303.17651_self_refine_chunk_016.pdf: ### Concepts: [ 'animal', 'catch', 'horse', 'lasso', 'ride'] Sentence: The horse catches the lasso and rides on it. what concepts from the concept list are missing from the sentence? Concept Feedback: animal Any feedback on commonsense? Commonsense Feedback: T
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_017.pdf: We want to iteratively improve the provided responses. To help improve, scores for each response on desired traits are provided: 1) Relevant, 2) Inf ormative, 3) Interesting, 4) Consistent, 5) Helpful, 6) Engaging, 7) Specific, 8) Safe, 9) User understanding, 
  - metrics_mentioned: em
- 2303.17651_self_refine_chunk_018.pdf: Very positive: If you 're looking for a truly magical experience in Vegas, look no further than the Trop! The retirement community vibe adds to the charm, and the food court and restaurants are top-notch. The free Folies Bergere show is a real treat and the ro
  - metrics_mentioned: em

## 2305.10601 - Tree of Thoughts: Deliberate Problem Solving with Large Language Models

### Skim (Chunk 1)
Tree of Thoughts: Deliberate Problem Solving with Large Language Models Shunyu Yao Princeton University Dian Yu Google DeepMind Jeffrey Zhao Google DeepMind Izhak Shafran Google DeepMind Thomas L. Griffiths Princeton University Yuan Cao Google DeepMind Karthik Narasimhan Princeton University Abstract Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, “Tree of Thoughts” (ToT), which generalizes over the popular “Chain of Thought” approach to prompting language models, and enables exploration over coherent units of text 

## 2309.02144 - Making Large Language Models Better Reasoners with Alignment

### Skim (Chunk 1)
Preprint MAKING LARGE LANGUAGE MODELS BETTER REA- SONERS WITH ALIGNMENT Peiyi Wang1 Lei Li3 Liang Chen1 Feifan Song1 Binghuai Lin2 Yunbo Cao2 Tianyu Liu2 Zhifang Sui1 1 National Key Laboratory for Multimedia Information Processing, Peking University 2 Tencent Cloud AI 3 The University of Hong Kong {wangpeiyi9979, nlp.lilei }@gmail.com leo.liang.chen@outlook.com; songff@stu.pku.edu.cn {binghuailin, yunbocao, rogertyliu }@tencent.com; szf@pku.edu.cn ABSTRACT Reasoning is a cognitive process of using evidence to reach a sound conclusion. The reasoning capability is essential for large language models (LLMs) to serve as the brain of the artificial general intelligence agent. Recent studies reveal that fine-tuning LLMs on data with the chain of thought (COT) reasoning process can significantly enhance their reasoning capabilities. However, we find that the fine-tuned LLMs suffer from an Asses

## 2312.08935 - Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations

### Skim (Chunk 1)
MATH-S HEPHERD : V ERIFY AND REINFORCE LLM S STEP -BY-STEP WITHOUT HUMAN ANNOTATIONS Peiyi Wang1† Lei Li3 Zhihong Shao4 R.X. Xu2 Damai Dai1 Yifei Li5 Deli Chen2 Y. Wu2 Zhifang Sui1 1National Key Laboratory for Multimedia Information Processing, Peking University 2DeepSeek-AI 3The University of Hong Kong 4Tsinghua University 5The Ohio State University {wangpeiyi9979, nlp.lilei }@gmail.com li.14042@osu.edu szf@pku.edu.cn Project Page: MA T H-SH E P H E R D ABSTRACT In this paper, we present an innovative process-oriented math process reward model called MATH-SHEPHERD , which assigns a reward score to each step of math problem solutions. The training of MATH-SHEPHERD is achieved using automati- cally constructed process-wise supervision data, breaking the bottleneck of heavy reliance on manual annotation in existing work. We explore the effectiveness of MATH-SHEPHERD in two scenarios: 1) Ve

## 2403.09629 - Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

### Skim (Chunk 1)
Quiet-ST aR: Language Models Can T each Themselves to Think Before Speaking Eric Zelikman Stanford University Georges Harik Notbad AI Inc Yijia Shao Stanford University V aruna Jayasiri Notbad AI Inc Nick Haber Stanford University Noah D. Goodman Stanford University Abstract When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the Self-Taught Reasoner (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting – ideally, a 

### Deep Read Notes (all chunks)
- 2403.09629_quiet_star_chunk_001.pdf: Quiet-ST aR: Language Models Can T each Themselves to Think Before Speaking Eric Zelikman Stanford University Georges Harik Notbad AI Inc Yijia Shao Stanford University V aruna Jayasiri Notbad AI Inc Nick Haber Stanford University Noah D. Goodman Stanford Univ
  - datasets_mentioned: CommonsenseQA, GSM8K
  - metrics_mentioned: Accuracy, accuracy, em
- 2403.09629_quiet_star_chunk_002.pdf: Algorithm 1: Quiet Self-Taught Reasoner (Quiet-STaR) Input: Language model θ0, training steps num steps, sequence length l, thought length t, learning rate α, batch size b, number of thoughts nthoughts , number of ground truth tokens used for supervising each 
  - metrics_mentioned: em
- 2403.09629_quiet_star_chunk_003.pdf: d e f g h Thought </t> g h <t> Thought Thought Thought Figure 4: Forward Pass and T eacher Forcing. We visualize a single forward pass of our algorithm. Solid lines denote language model computation, while dashed lines indicate tokens are inserted via teacher 
  - datasets_mentioned: CommonsenseQA, GSM8K
  - metrics_mentioned: Accuracy, accuracy, em
- 2403.09629_quiet_star_chunk_004.pdf: '<s> # Magnesium reacts with nitrogen to form magnesium nitride. The chemical formula for this reaction is Mg+N_2-> MgN_2. What is the product, or what are the products, of this reaction?\n\nJan 12, 2016\n\nThe formula for magnesium nitride is $M {g}_{3} {N}_{
  - datasets_mentioned: CommonsenseQA
  - metrics_mentioned: em
- 2403.09629_quiet_star_chunk_005.pdf: Michael Y Li, Emily B Fox, and Noah D Goodman. Automated statistical model discovery with language models. arXiv preprint arXiv:2402.17879, 2024. Shiyang Li, Jianshu Chen, Yelong Shen, Zhiyu Chen, Xinlu Zhang, Zekun Li, Hong Wang, Jing Qian, Baolin Peng, Yi Ma
  - datasets_mentioned: MATH
  - metrics_mentioned: EM, Em, em
- 2403.09629_quiet_star_chunk_006.pdf: Appendix A Hyperparameter Choices Optimization and Evaluation For optimization, we use the AdamW optimizer with a warmup of 20 steps, a learning rate of 1e − 6, a weight decay of 0.001, and a batch size of 8 (along with any necessary gradient accumulation to k
  - datasets_mentioned: CommonsenseQA, GSM8K
  - metrics_mentioned: Accuracy, accuracy, em
- 2403.09629_quiet_star_chunk_007.pdf: Moving from the top line to the second line, we multiply the numbers within each parenthetical group. Moving from the second line to the third line, we add the products together to find the total. Finally, we can distribute and determine the final product: $= 
  - metrics_mentioned: em
- 2403.09629_quiet_star_chunk_008.pdf: A: Let 's think step by step. 1. The ducks lay 16 eggs per day. 2. She eats 3 for breakfast every morning. 3. She bakes muffins for her friends every day with 4. 4. She sells the remainder at the farmers ' market daily for $2 per fresh duck egg. 5. The number 
  - metrics_mentioned: em
- 2403.09629_quiet_star_chunk_009.pdf: H Contribution Visualization 0 icos ^ 2 θ sin ^ 3 θ + 5 cos θ sin ^ 4 θ + is in ^ 5 θ $ Then I used the Mo iv re ' s theorem and I got : $( cos 5 θ + is in 5 θ )$ I compared the imaginary parts and I got something like : $ sin 5 θ = 5 cos ^ 4 θ sin θ - 1 0 cos
  - metrics_mentioned: em

## 2506.18167 - Understanding Reasoning in Thinking Language Models via Steering Vectors

### Skim (Chunk 1)
Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs UNDERSTANDINGREASONING INTHINKINGLAN- GUAGEMODELS VIASTEERINGVECTORS Constantin V enhoff∗ University of Oxford United Kingdom constantin@robots.ox.ac.uk Iv´an Arcuschin∗ University of Buenos Aires Argentina iarcuschin@dc.uba.ar Philip Torr University of Oxford United Kingdom Arthur Conmy Neel Nanda ABSTRACT Recent advances in large language models (LLMs) have led to the development of thinkinglanguage models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, control- ling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models. Through a systematic experiment on 500 tasks across 10 diverse categories, we identify

### Deep Read Notes (all chunks)
- 2506.18167_reasoning_via_steering_vectors_chunk_001.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs UNDERSTANDINGREASONING INTHINKINGLAN- GUAGEMODELS VIASTEERINGVECTORS Constantin V enhoff∗ University of Oxford United Kingdom constantin@robots.ox.ac.uk Iv´an Arcuschin∗ University of Buenos Ai
  - datasets_mentioned: ARC
  - metrics_mentioned: EM, em
- 2506.18167_reasoning_via_steering_vectors_chunk_002.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs Backtracking Uncertainty Estimation Example T esting Adding Knowledge Initializing Deduction 0% 10% 20% 30% 40% 50% 60%Avg Sentence Fraction Avg: 4% Avg: 9% Avg: 6% Avg: 15% Avg: 7% Avg: 52% Av
  - metrics_mentioned: em
- 2506.18167_reasoning_via_steering_vectors_chunk_003.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs backtrackinguncertaintyestimation exampletestingadding knowledge 0% 10% 20% 30% 40% 50% 60% 70% Average Sentence Fraction (%) DeepSeek-R1-Distill-Llama-8B Original Positive Steering Negative St
  - datasets_mentioned: ARC
  - metrics_mentioned: EM, accuracy, em
- 2506.18167_reasoning_via_steering_vectors_chunk_004.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs Mike Knoop. R1-Zero and R1 Results and Analysis, January 2025. URL https://arcprize. org/blog/r1-zero-r1-results-analysis. Kenneth Li, Oam Patel, Fernanda Vi ´egas, Hanspeter Pfister, and Marti
  - metrics_mentioned: Em, em, f1
- 2506.18167_reasoning_via_steering_vectors_chunk_005.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs Automatically annotated response: (colored by assigned label) ["initializing"]Okay, so I came across this riddle. At first glance, it seems tricky, but I can break it down.["end-section"] ["ded
  - metrics_mentioned: EM, Em, em
- 2506.18167_reasoning_via_steering_vectors_chunk_006.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs ["uncertainty-estimation"] Hmm, let’s see. ["end-section"] ["adding-knowledge"] I remember that probability problems often involve combinations, ["end-section"] ["deduction"] so maybe I should 
  - metrics_mentioned: em
- 2506.18167_reasoning_via_steering_vectors_chunk_007.pdf: Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs First, I need to determine the number of ways to draw exactly two hearts from the deck. ["end-section"] ["adding-knowledge"] There are 13 hearts in the deck, and I want to choose 2 of them. ["e
  - metrics_mentioned: em

## 2512.07667 - Depth-Wise Activation Steering for Honest Language Models

### Skim (Chunk 1)
Depth-Wise Activation Steering for Honest Language Models Gracjan Góral∗ University of Warsaw MARS Marysia Winkels∗ MARS Steven Basart Center for AI Safety Abstract Large language models sometimes assert falsehoods despite internally representing the correct answer—failures of honesty rather than accuracy—which undermines auditability and safety. Existing approaches largely optimize factual correctness or depend on retraining and brittle single-layer edits, offering limited leverage over truthful reporting. We present a training-free activation steering method that weights steering strength across network depth using a Gaussian schedule. On the MASK benchmark—which separates honesty from knowledge—we evaluate seven models spanning the LLaMA, Qwen, and Mistral families and find that Gaussian scheduling improves honesty over no-steering and single-layer baselines in six of seven models. Eq
