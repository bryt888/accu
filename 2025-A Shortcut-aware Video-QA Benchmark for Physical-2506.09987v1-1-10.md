# A Shortcut-aware Video-QA Benchmark for Physical Understanding via Minimal Video Pairs 

Benno Krojer ${ }^{1,2,3, *}$, Mojtaba Komeili ${ }^{1}$, Candace Ross ${ }^{1}$, Quentin Garrido ${ }^{1}$, Koustuv Sinha ${ }^{1}$, Nicolas Ballas ${ }^{1}$, Mahmoud Assran ${ }^{1}$<br>${ }^{1}$ FAIR at Meta, ${ }^{2}$ Mila, ${ }^{3}$ McGill University<br>*Work done during internship


#### Abstract

Existing benchmarks for assessing the spatio-temporal understanding and reasoning abilities of video language models are susceptible to score inflation due to the presence of shortcut solutions based on superficial visual or textual cues. This paper mitigates the challenges in accurately assessing model performance by introducing the Minimal Video Pairs (MVP) benchmark, a simple shortcut-aware video QA benchmark for assessing the physical understanding of video language models. The benchmark is comprised of 55 K high-quality multiple-choice video QA examples focusing on physical world understanding. Examples are curated from nine video data sources, spanning first-person egocentric and exocentric videos, robotic interaction data, and cognitive science intuitive physics benchmarks. To mitigate shortcut solutions that rely on superficial visual or textual cues and biases, each sample in MVP has a minimal-change pair - a visually similar video accompanied by an identical question but an opposing answer. To answer a question correctly, a model must provide correct answers for both examples in the minimal-change pair; as such, models that solely rely on visual or textual biases would achieve below random performance. Human performance on MVP is $92.9 \%$, while the best open-source state-of-the-art video-language model achieves $40.2 \%$ compared to random performance at $25 \%$.


Date: June 12, 2025
Correspondence: Benno Krojer at benno.krojer@mila.quebec

## 1 Introduction

Moravec's paradox highlights a counterintuitive phenomenon: high-level reasoning tasks, often perceived as complex, are typically easier for AI agents to solve than sensorimotor and perception tasks, which are seemingly effortless for humans (Moravec, 1988).

Recently, large vision-language models have emerged as a promising paradigm for enabling perception capabilities in AI agents, demonstrating impressive progress on question-answering tasks across various domains including movies, documents, charts, and sciences (Alayrac et al., 2022; Team et al., 2024; Dubey et al., 2024; Wang et al., 2024a). This progress raises a natural question: do these models possess the spatiotemporal understanding and reasoning abilities essential for an agent to interact within the physical world, or do they buttress Moravec's paradox?

Various visual QA datasets have been proposed by the community to assess the spatiotemporal understanding of video-language models (Tapaswi et al., 2016; Maharaj et al., 2017; Li et al., 2024b; Patraucean et al., 2023; Zhang et al., 2023c; Xie et al., 2025; Wang et al., 2023c; Yi* et al., 2020); one of the most popular, MVBench (Li et al., 2024b), combines 11 video datasets into a single
video QA benchmark.
While recent state-of-the-art video-language models obtain performance far superior to a random baseline on these benchmarks (Wang et al., 2024a; Shen et al., 2024; Li et al., 2024a), our investigation reveals that existing models can achieve strong performance on these benchmarks by relying on superficial visual or textual cues or biases. This is validated using simple baselines that discard the visual input or temporal aspect, yet achieve non-trivial performance. Similarly, concurrent work (Cores et al., 2024) shows that some of these tasks (Li et al., 2024b) fail to accurately measure the temporal understanding of a model.

In this work, we take inspiration from works in natural language processing (Levesque et al., 2012; Sakaguchi et al., 2021) and image processing (Thrush et al., 2022; Yuksekgonul et al., 2022) addressing visual and textual biases in evaluation, and introduce MVP, a video QA benchmark containing minimal-change video pairs. Specifically, each video-question-answer sample in the benchmark is accompanied by a visually similar video possessing an identical question but an opposing answer (Figure 2). To answer a question correctly, a model must also provide the correct answer for its minimal-change pair while processing them independently. Many types
![](https://cdn.mathpix.com/cropped/2025_06_19_904608b980ffcac2d6a6g-02.jpg?height=966&width=1703&top_left_y=178&top_left_x=203)

Figure 1 Illustrating MVP with its curation steps (top) and examples of our Minimal Pair Scoring (bottom).
of shortcut solutions are penalized under the minimalpair scoring framework as a model relying on superficial visual or textual cues or biases would incorrectly output the same answer for both the samples in the pair.

While recent work created small sets of minimal-change video pairs for course-grained temporal reasoning (Zhang et al., 2024a; Liu et al., 2024), our key insight is that these pairs can be efficiently mined from existing video sources to test for several model capabilities through an automated process relying on visual embeddings and video meta-data. We propose an automatic process to find minimally different video pairs with limited human intervention, and then build these into a video-questionanswer tuple with identical questions and opposing answers, enabling the scaling of the benchmark to a broad set of videos spanning diverse situations. We further process the mined samples using a model ensemble to filter out single-frame solvable examples - questions that can be answered using any single randomly sampled frame from the video - to encourage a stronger focus on video understanding. We build MVP by running our process on nine video sources spanning intuitive physics understanding, spatiotemporal reasoning, action anticipation, and robotic manipulation, leading to a total of 54,828 multiple-choice video QA examples with minimal-change pairs; i.e. 27, 414 minimal-change pairs.

Next, we assess recent proprietary and open-source state-of-the-art video-language models using MVP.

Specifically, we evaluate 2 closed-source models (GPT4o (Achiam et al., 2023) and Gemini-1.5 Pro (Team et al., 2024)), and 7 open-source video-language models: LLaVA-OneVision Li et al. (2024a), VideoChat2 Li et al. (2024b), Mini-CPM (Yao et al., 2024), Qwen2VL (Bai et al., 2023), Tarsier (Wang et al., 2024a), LongVu (Shen et al., 2024), InternVL-2.5 (Chen et al., 2024b). We find that even proprietary models are only slightly above random and that the best accuracy achieved across models is only $40.2 \%$, in stark contrast to human baseline performance at $92.9 \%$ accuracy. These findings suggest that video-language models may still struggle with seemingly simple physical reasoning tasks, despite achieving relatively high accuracy on standard spatio-temporal reasoning benchmarks.

In short, we make the following contributions:

1. Analyze potential shortcut solutions on all 11 datasets in the popular MVBench (Li et al., 2024b) benchmark suite, using simple baselines consisting of language-only models, single-frame/image models, and Socratic LLMs.
2. Introduce MVP, a video QA benchmark for physical world understanding comprising minimally different videos - the largest of its kind by an order of magnitude with $\sim 55 \mathrm{~K}$ examples.
3. Benchmark closed-source and open-source state-
![](https://cdn.mathpix.com/cropped/2025_06_19_904608b980ffcac2d6a6g-03.jpg?height=579&width=838&top_left_y=182&top_left_x=199)

Figure 2 Performance of the strongest evaluated VideoLLMs on MVP (mini-version), compared to human performance.
of-the-art models and identify a gap in physical world understanding; human performance on MVP is around $92.9 \%$, while even GPT4-o and Gemini achieve around $30 \%$ compared to random performance at $25 \%$.

## 2 Robustness Analysis of MVBench

We begin by examining robustness of existing video QA benchmarks to shortcut solutions based on visual or textual cues or biases. Specifically, our analysis focuses on CLEVRER (Johnson et al., 2017), Perception Test (Patraucean et al., 2023), STAR (Wu et al., 2021), PAXION (Wang et al., 2023c), Moments in Time V1 (Monfort et al., 2020), FunQA (Xie et al., 2025), CharadesSTA (Gao et al., 2017), MoVQA (Zhang et al., 2023b), NTU RGB+D (Liu et al., 2020), VLN-CE (Krantz et al., 2020) and TVQA (Lei et al., 2018), which are all included in the widely adopted MVBench (Li et al., 2024b) benchmark suite.

Empirical Setup. MVBench is comprised of 20 tasks from 11 datasets, collected in a multiple-choice video QA format, where a model is required to choose an answer $a_{i}$ from a tuple of question, video, and answer candidates $\left(q, v,\left[\mathrm{a}_{1}, \mathrm{a}_{2}, ..\right]\right)$. Following standard practice (Goyal et al., 2017), we study robustness to shortcuts by perturbing the task inputs, e.g., requiring the model to select an answer candidate without seeing the video or perhaps without reading the question, and compare to the accuracy achieved by a video LLM without perturbing the task inputs. We study 4 types of shortcut solutions by evaluating language-only models, video-only models, single-frame models, and simple Socratic LLMs. Results are reported in Table 1 using the original skill taxonomy outlined in MVBench.

Language only. Language-only models do not observe the video, and therefore select an answer candidate
by only considering the textual inputs $q$ and the answer candidates $\left[\mathrm{a}_{1}, \mathrm{a}_{2}, ..\right]$. We leverage the Llama3-8B and Llama3-70B models due to their competitive performances (Dubey et al., 2024). In Table 1, we find that a Llama3-8B outperforms a random baseline by $6 \%$, and a larger Llama3-70B outperforms a random baseline by $8 \%$, suggesting that only a small subset of examples can be solved without considering the video input. However, digging into the individual datasets and sub-tasks in Table 1 reveals strong language-only performance on Action Antonym, where LLaMA3-70 achieves $78 \%$ compared to a random baseline at $50 \%$. Upon closer inspection of the original dataset, we observe that many questions can be correctly selected by choosing the answer candidate with the highest marginal likelihood. For instance, given an example with answer candidates "book falling like a rock" versus "book rising like a rock," an LLM, just like a human, can rely on its language bias to infer that the former is probably the correct description without observing the video.

Video only. Video-only models do not observe the question, and therefore select an answer candidate by only considering the video input $v$ and the answer candidates $\left[\mathrm{a}_{1}, \mathrm{a}_{2}, ..\right]$. Table 1 shows that a video LLM (VideoChat2Mistral) can solve most sub-tasks without access to the question, reaching 50\% overall accuracy; by comparison the same model achieves an accuracy of $61 \%$ when given the question in addition to the video, while a random baseline is at $30 \%$. These findings indicate that the answer candidates for each question $\left[a_{1}, a_{2}, ..\right]$ are not sufficiently task-specific, as the model is able to discard the incorrect answers without knowing question.

This trend is particularly interesting on the counterfactual inference sub-task, where the counterfactual scenario such as "What happens if the cube is removed?" can only be known from the question. Manual inspection reveals that the correct answer in this task (based on CLEVRER (Yi* et al., 2020)) often occurs regardless of the counterfactual scenario, e.g., the two objects in question will collide regardless of the causal intervention.

Single-frame only. Single-frame models do not observe the entire video, but rather are provided only a single frame $f_{i} \in v$ form the video. These models must therefore select an answer candidate by considering the frame $f_{i}$, the textual inputs $q$, and the answer candidates $\left[\mathrm{a}_{1}, \mathrm{a}_{2}, ..\right]$. We take $f_{i}$ to be the center frame from the video and leverage Idefics3-8B (Laurençon et al., 2024) and Qwen2-VL-7B (Wang et al., 2024b) for the single-frame baselines. In Table 1, Idefics3-8B achieves an overall accuracy of $47 \%$ and Qwen2-VL-7B achieves an overall $51 \%$ accuracy, which is comparable to the performance of full-fledged VideoLLMs. Moreover, on Action Antonym, Action Prediction, Character Order, Egocentric Navigation, Episodic Reasoning, Fine-grained Action, State Transition, and Unexpected Action, the single-frame models are on par with (or even exceed) the

| Task | Avg | AA | AC | AL | AP | AS | CI | CO | EN | ER | FA | FP | MA | MC | MD | OE | OI | OS | ST | SC | UA |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Random Chance | 0.30 | 0.50 | 0.33 | 0.25 | 0.25 | 0.25 | 0.31 | 0.33 | 0.25 | 0.20 | 0.25 | 0.25 | 0.33 | 0.25 | 0.25 | 0.50 | 0.25 | 0.33 | 0.25 | 0.33 | 0.25 |
| GPT-4V ${ }^{\dagger}$ | 0.44 | 0.72 | 0.39 | 0.41 | 0.64 | 0.56 | 0.11 | 0.52 | 0.31 | 0.59 | 0.47 | 0.48 | 0.23 | 0.12 | 0.12 | 0.19 | 0.59 | 0.30 | 0.84 | 0.45 | 0.74 |
| VideoChat2 (Mistral) | 0.61 | 0.86 | 0.37 | 0.44 | 0.55 | 0.76 | 0.72 | 0.49 | 0.36 | 0.40 | 0.50 | 0.64 | 0.88 | 0.69 | 0.49 | 0.87 | 0.75 | 0.41 | 0.85 | 0.50 | 0.62 |
| Language only: Model considers question and answer choices, without access to the video. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Llama 3-8B | 0.36 | 0.63 | 0.38 | 0.27 | 0.25 | 0.28 | 0.35 | 0.43 | 0.29 | 0.43 | 0.29 | 0.29 | 0.38 | 0.27 | 0.21 | 0.46 | 0.29 | 0.36 | 0.52 | 0.40 | 0.52 |
| Llama 3-70B | 0.38 | 0.78 | 0.39 | 0.32 | 0.26 | 0.26 | 0.43 | 0.47 | 0.28 | 0.46 | 0.26 | 0.27 | 0.41 | 0.29 | 0.20 | 0.48 | 0.29 | 0.32 | 0.48 | 0.45 | 0.58 |
| Video only: Model considers video and answer choices only, without access to the question. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| VideoChat2 (Mistral) | 0.50 | 0.88 | 0.42 | 0.25 | 0.49 | 0.68 | 0.74 | 0.44 | 0.28 | 0.39 | 0.53 | 0.65 | 0.47 | 0.29 | 0.26 | 0.53 | 0.75 | 0.34 | 0.81 | 0.32 | 0.55 |
| Single-Frame only: Model considers question, answer choices and a single key frame, without access to the full video. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Idefics3 | 0.47 | 0.72 | 0.37 | 0.31 | 0.52 | 0.48 | 0.42 | 0.54 | 0.31 | 0.48 | 0.40 | 0.44 | 0.55 | 0.42 | 0.34 | 0.49 | 0.50 | 0.37 | 0.73 | 0.48 | 0.60 |
| Qwen2-VL | 0.51 | 0.87 | 0.37 | 0.31 | 0.55 | 0.54 | 0.57 | 0.59 | 0.40 | 0.45 | 0.46 | 0.53 | 0.6 | 0.43 | 0.37 | 0.53 | 0.54 | 0.39 | 0.74 | 0.42 | 0.68 |
| Simple Socratic LLM: Model considers the question, answer choices and a short generic description of the video. |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Llama 3-8B | 0.44 | 0.56 | 0.38 | 0.28 | 0.49 | 0.57 | 0.35 | 0.53 | 0.29 | 0.42 | 0.30 | 0.35 | 0.56 | 0.42 | 0.32 | 0.50 | 0.56 | 0.35 | 0.68 | 0.44 | 0.56 |
| Llama 3-70B | 0.46 | 0.67 | 0.32 | 0.35 | 0.40 | 0.55 | 0.38 | 0.55 | 0.24 | 0.45 | 0.36 | 0.41 | 0.56 | 0.46 | 0.32 | 0.57 | 0.62 | 0.35 | 0.70 | 0.39 | 0.54 |

Table 1 Shortcut Analysis on the 20 MVBench tasks from 11 datasets: Optimal performance on these spatio-temporal reasoning benchmarks is frequently achieved by models relying on visual or textual biases (Single-Frame only, Video only, Simple Socratic LLM). ${ }^{\dagger}$ : GPT-4V accuracy from (Li et al., 2024b). Tasks: AA (Action Antonym), AC (Action Count), AL (Action Localization), AP (Action Prediction), AS (Action Sequence), CI (Counterfactual Inference), CO (Character Order), EN (Egocentric Navigation), ER (Episodic Reasoning), FA (Fine-grained Action), FP (Fine-grained Pose), MA (Moving Attribute), MC (Moving Count), MD (Moving Direction), OE (Object Existence), OI (Object Interaction), OS (Object Shuffle), ST (Scene Transition), SC (State Change), UA (Unexpected Action).
performance of the VideoLLMs. Concurrent work (Cores et al., 2024) also studies the related bag-of-frame bias by shuffling the video frames.

Simple Socratic LLM. A Simple Socratic LLM (Zhang et al., 2023a; Zeng et al., 2023) replaces the video input $v$ with a short caption $c_{v}$ that can only convey a lowbandwidth description of the video. In practice, $c_{v}$ is 1 or 2 sentence-long caption generated by a separate VideoLLM (Zhang et al., 2024b) in a task-independent manner. The Socratic LLMs therefore select an answer candidate by only considering the low-bandwidth caption $c_{v}$, the question $q$, and the answer candidates $\left[\mathrm{a}_{1}, \mathrm{a}_{2}, ..\right]$. Following the text-only baselines, we use Llama3-8B and 70B. The performance of the Simple Socratic LLMs in Table 1 is significantly above random, with $44 \%$ for the LLaMA3-8B and $47 \%$ for the LLaMA-70B, suggesting that many sub-tasks (e.g. Character Order, Episodic Reasoning, Scene Transition) do not require fine-grained scene understanding.

Summary. The shortcut analysis reveals that existing models can often achieve strong performance on spatiotemporal reasoning benchmarks by relying on language cues (Language only shortcut) or visual cues, (Video only shortcut), and may not need to perform temporal reasoning (Single-Frame only shortcut), or possess finegrained visual features (Simplified Socratic LLM).

## 3 Testing Physical World Understanding via Minimal Change Pairs

In this section we discuss the construction of MVP to mitigate shortcut solutions based on visual and textual biases. MVP is comprised of 54,828 video QA examples covering various aspects of physical world understand-
ing, including spatial reasoning, temporal understanding, human-object interaction, memory, counterfactuals, anticipation, and intuitive physics.

Task formulation. To improve robustness to the various shortcut solutions described in the previous section, we adopt a minimal-change pair approach (Levesque et al., 2012; Sakaguchi et al., 2021). An example in MVP consists of two video QA pairs ( $q_{1}, v_{1},\left[\mathrm{a}_{1}, \mathrm{a}_{2}\right]$ ) and ( $q_{2}, v_{2},\left[\mathbf{a}_{1}, \mathbf{a}_{2}\right]$ ) containing identical questions $q_{1}=q_{2}$, visually similar videos $v_{1} \sim v_{2}$, and two mutually exclusive (i.e., contradicting) answer candidates $a_{1}$ and $a_{2}$.

Minimal-change Pair Scoring. A model relying on superficial visual or textual cues or biases to solve a task will tend to produce the same output for each sample in the minimal-change pair. Thus, to penalize models for latching onto shortcuts, we only provide a positive score if the correct answer is produced for both minimal-change samples; the model receives each example ( $q, v_{1},\left[\mathrm{a}_{1}, \mathrm{a}_{2}\right]$ ) and ( $q, v_{2},\left[\mathrm{a}_{1}, \mathrm{a}_{2}\right]$ ) in isolation. Following a multiple choice QA framework, the model has to output a single answer letter (A or B) via task-specific prompts. In this setup, a random baseline achieves an accuracy of $25 \%$.

Question Taxonomy. We wish to understand whether video LLMs possess the spatio-temporal understanding and reasoning abilities essential for an agent to interact within the physical world. As such we consider a coarsegrained taxonomy of question categories encompassing:

- Fine-grained human-object interactions,
- Fine-grained robot-object interactions,
- Intuitive Physics understanding,
- Coarse-grained temporal understanding.

We intentionally construct samples that are not overly

| Benchmark | Domains of VideoQA-Examples |  |  |  |  | Minimially Diff. Videos | Procedural Single-Frame Bias Filtering | Format |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
|  | Total | Natural Videos | Intuitive Physics | Robotics | Synthetic Videos |  |  |  |
| CLEVRER | 76.3 K | 0K | 21.4 K | 0K | 76.3 K | $\times$ | $\times$ | MC-QA |
| Perception Test | 11.5 K | 11.5 K | $0-0.2 \mathrm{~K}$ | 0K | 0K | $\times$ | $\times$ | MC-QA |
| MVBench | 4 K | 2.8 K | 0K | 0.2 K | 1.2 K | $\times$ | $\times$ | MC-QA |
| TVBench | 2.5 K | 1.9 K | 0K | 0.2 K | 0.6K | $\times$ | $\times$ | MC-QA |
| Vinoground | 1 K | 1 K | 0K | 0K | 0K | $\checkmark$ | $\times$ | Group-Score |
| TempCompass | 0.5 K | 0.5 K | 0K | 0K | 0K | $\checkmark$ | $\times$ | Group-Score |
| MVP | 54.8 K | 22.3 K | 9.9 K | 25.8 K | 32.6 K | $\checkmark$ | $\checkmark$ | Pair MC-QA |

Table 2 We compare with recent benchmarks that focus on similar skills. Note that some videos may fall within several categories (e.g., synthetic intuitive physics videos). MVP contains minimally different videos at a much larger scale and across more diverse domains. From these benchmarks, MVP is the first to procedurally filter out examples due to single-frame bias. Group-Score $=$ Present one video + two captions, and two videos + one caption. CLEVRER's intuitive physics entry is grayed as it only covers a narrow subset of intuitive physics concepts, largely based on collisions.

| Benchmark Category | Sources (\# paired video-QA examples) | Example |
| :--- | :--- | :--- |
| Fine-grained human-object interactions | Perception Test (3.5K), Something Something v2 (3.6K) | Q: What stops the motion of the object placed on the slanted plane after being released [...]? A) Person or collision with another object B) High friction with surface |
| Fine-grained robot-object interactions | Language Table (12.9K) | Q: Which robot instruction best describes the actions in the video? A) Move the green blocks in a vertical line below blue cube B) Move the green blocks and blue cube in a vertical line |
| Intuitive physics and collisions | IntPhys (0.2K), InfLevel (2.6K), GRASP (2.0K), CLEVRER (1.2K) | Q: Is this video physically plausible/possible according to your understanding of e.g. object permanence, gravity, [...] A) Yes, everything is behaving according to human intuitive physics understanding B) No, something in the video is off/strange or violates [...] |
| Coarse-grained temporal reasoning | STAR (1.0K), Vinoground (0.5K) | Q: What is the best caption for this video? A) The kayak flips over from facing upwards towards facing downwards B) The kayak flips over from facing downwards towards facing upwards |

Table 3 Overview of MVP. Each answer option A/B is correct for only one video in the minimal-change pair, while acting as a hard negative for the other video. Note that we show the number of paired video-QA examples, thus the number of videos in our data is twice that amount.
reliant on cultural knowledge (Rawal et al., 2024; He et al., 2024; Li et al., 2024c) (e.g., movies) or specific domain knowledge (Tang et al., 2019) (e.g., detailed recipes) - tasks where language bias could contribute to the general performance.

We first manually filter videos from the sources described in Table 3 based on manual inspection (cf. Appendix B.1), then convert them into a question-answer format based on the associated meta-data (the textual captions for Language Table, the class labels for Something-Something-v2, QA annotations for PT, Vinoground, STAR, and CLEVERER, and the concept labels for IntPhys, InfLevel, and GRASP), yielding a starting set of 548 K video QA examples.

Minimal-change Pair Mining. Next we procedurally identify minimal-change pairs from the 548 K video QA ex-
amples produced from the previous stage. We note that $16 \%$ of the videos in our final benchmark ( $\sim 8.8 \mathrm{~K}$ examples) already possess explicit minimal visual pairs. For the remaining $84 \%$ of the videos, we leverage the following procedure to construct visual minimal-change pairs. In this process, we search for samples that have visually similar videos, identical questions (based on semantic matching), and contradictory answers. To then determine whether two videos with the same question are suitable minimal pairs, we use a) symbolic and neural rules to determine video similarity and b) entailment detection (Bowman et al., 2015; Dagan et al., 2013) between the correct answers of each video. Whether we rely more on symbolic or or neural rules of similarity depends on the data source: If a dataset has rich annotations (positions or attributes of objects) or structured captions (such as CLEVRER or Something Something-v2), we use
hand-crafted rules and the NLP toolkit spacy (Honnibal and Montani, 2017) to narrow down the candidate pool of minimal pairs. This step would match videos with a large intersection of objects or attributes mentioned in the annotation/caption, leading to highly similar videos (e.g., the same objects appearing in both videos). Once we have narrowed down the pool of candidate pairs, in the final step we rank video pairs by their cosine similarity in the ViCLIP (Wang et al., 2023b) video embedding space. We then select the top-ranked minimal video pairs such that each question or skill-type is sufficiently represented. At the same time, we ensure that the correct answers for samples in a minimal-change pair are sufficiently different, as the correct answer of one element in the pair must be a truly negative (negative) answer candidate for the other element, and vice versa: To avoid cases where both answers could be true at the same time (e.g., synonyms or more subtle cases) we define a set of textual rules to detect entailment for a subset of datasets. To illustrate this, in the Fine-grained Robot-object interactions category, our entailment-detection would discard the following pair of answers: A) "Move the blue cube towards the red heart" and B) "Move the blue cube to the left of the red heart", since A entails B. After this minimal-pair mining, we are down to 70 K QA examples; cf. Appendix B. 2 for technical details of the minimal pair mining process.

Single-frame Bias Filtering. Finally, to address singleframe bias, we remove examples that can be solved without the temporal information in the video; i.e., using only a single frame. We note that the input frame for this filtering stage should not be selected in a "smart way," since key-frame selection can be regarded as a basic form of temporal reasoning. In practice, five state-of-the-art multi-modal LLMs (LLama3.2-11B (Dubey et al., 2024), Molmo-7B (Deitke et al., 2024), Pixtral-12B (Agrawal et al., 2024), LLaVA-OneVision-7B (Li et al., 2024a), Idefics3-8B (Laurençon et al., 2024)) are prompted to answer the video-QA questions and "give their best commonsense guess given a single frame sampled from a video." If at least 4 out of 5 models in the ensemble predict the correct answer given the same frame, then we flag that frame as solvable. The minimal-change pair is then discarded if $30 \%$ of the frames in both videos are deemed solvable. This heuristic process removes around $20 \%$ of the samples from the previous stage.

MVP Statistics. We end up with 54,828 examples in MVP, grouped into 27, 414 minimal-change video QA pairs. A breakdown of these examples is shown in Table 2 and Table 3 with a reasonably balanced split between natural videos, synthetic videos, robotics videos, and intuitive physics videos. An average video is 8.8 seconds long, the answer candidates contain an average of 8.1 words, and the datasets contains 2355 unique words in the questions and answers. Note that the word diversity is much less than MVBench (Li et al., 2024b), which
has only 4 K examples but twice the number of unique words (4338), reflecting our focus in testing for physical world understanding and not linguistically-diverse tasks with cultural or domain knowledge. Instead the task difficulty arises from the physical and perceptual aspects of MVP.

## 4 Empirical Results on MVP

We evaluate several state-of-the-art open-source VideoLLMs on MVP, summarized in Table 4: LLaVaOneVision (Li et al., 2024a), VideoChat2 (Li et al., 2024b), Mini-CPM-v 2.6 (Yao et al., 2024), Qwen2VL (Wang et al., 2024b), Tarsier (Wang et al., 2024a) 7B/34B, LongVU (Shen et al., 2024), InternVL2.58B (Chen et al., 2024b), Gemini-1.5 Pro (Team et al., 2024), and GPT4-o (Achiam et al., 2023). Most notably these models differ in their generality: The models we evaluate are either generalist models (GPT4-o, Gemini 1.5), specialized for any visual inputs (LLaVa-OneVision, Mini-CPM, Qwen2-VL, InternVL), or specialized primarily for videos (VideoChat2, LongVU). We also consider two baselines that are fed single-images, LLaVAOneVision and Qwen2-VL, as they have been trained to process both single image and video. Note that we additionally evaluate on a smaller balanced version of MVP, dubbed MVP-mini, with $1 / 3$ of the original size. ${ }^{1}$

Overall performance of VideoLLMs. Despite their strong performances on other video QA benchmarks (Li et al., 2024b; Liu et al., 2024; Mangalam et al., 2024; Xiao et al., 2021), Table 4 shows that most models perform around random chance ( $25 \%$ accuracy) with the exception of the Tarsier-34B model and InternVL2.5, reaching an average accuracy of $38.1 \%$ and 40.2 respectively. This is in contrast to human performances which obtain an average accuracy of $92.9 \%$ on a representative subset of MVP (cf. Appendix D).

While average performance is close to random for most models, we do observe non-trivial performance on several sub-tasks and data sources. In particular, VideoLLMs achieve better than random performance on Coarsegrained temporal reasoning, meaning they possess some ability to distinguish the order of events in a video.

All models fall short on Fine-grained robot-object interactions, which involves understanding fine-grained object manipulation on a table with a robotic arm. This is particularly interesting given the proliferated usage of multi-modal LLMs for learning large-scale visuomotor control policies (Driess et al., 2023; Jiang et al., 2023). Most notably, the Intuitive physics category of MVP is by far the hardest with sub-random scores. As highlighted by previous works, intuitive physics reasoning is known to be a difficult task (Riochet et al., 2022; Jassim

[^0]| Model | MVP (macro-avg) | Fine-grained human-object interactions | Fine-grained robot-object interactions | Intuitive physics and collisions | Coarse-grained temporal reasoning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Random | 25.0 (25.0) | 25.0 (25.0) | 25.0 (25.0) | 25.0 (25.0) | 25.0 (25.0) |
| Any text model ${ }^{\dagger}$ | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) | 0.0 (0.0) |
| Single-Frame Baseline (access to question, answer choices, and a single key frame from the video.) |  |  |  |  |  |
| LLaVA-OV (Qwen2-7B) | 11.8 (11.7) | 14.7 (12.2) | 8.7 (10.5) | 2.0 (2.3) | 21.6 (21.9) |
| Qwen2-VL (7B) | 16.7 (15.7) | 16.9 (13.6) | 20.1 (19.9) | 3.7 (4.4) | 26.3 (24.8) |
| VideoLLMs (full access to the video, question, and answer choices.) |  |  |  |  |  |
| LLaVA-OV (Qwen2-7B) | 20.7 (20.5) | 24.3 (21.8) | 5.2 (5.2) | 5.8 (6.8) | 47.5 (48.2) |
| VideoChat2 (Mistral-7B) | 23.3 (22.0) | 25.7 (21.0) | 21.4 (20.1) | 10.1 (11.5) | 35.8 (35.3) |
| Mini-CPM-v 2.6 | 21.7 (22.3) | 21.3 (20.2) | 18.0 (17.9) | 9.2 (11.9) | 38.3 (39.2) |
| Qwen2-VL (7B) | 30.0 (29.2) | 27.1 (32.28) | 27.6 (21.2) | 20.0 (18.9) | 45.2 (44.5) |
| LongVU (LLaMA3-3B) | 20.6 (20.6) | 15.8 (14.1) | 14.8 (16.0) | 16.2 (16.7) | 35.4 (35.8) |
| LongVU (Qwen2-7B) | 29.9 (29.3) | 28.9 (26.3) | 21.5 (21.8) | 20.5 (22.3) | 48.6 (46.7) |
| Tarsier-7B | 26.0 (24.3) | 31.3 (24.5) | 18.7 (18.2) | 15.0 (16.3) | 38.9 (38.2) |
| Tarsier-34B | 38.8 (37.4) | 45.2 (38.7) | 36.3 (36.6) | 21.0 (22.1) | 52.7 (52.4) |
| InternVL2.5-8B | 40.2 (39.9) | 43.7 (38.1) | 40.2 (38.7) | 22.8 (23.1) | 54.4 (59.8) |
| Gemini-1.5 Pro | - (29.6) | - (43.1) | - (15.5) | - (19.6) | - (40.2) |
| GPT4-o | - (32.5) | - (36.1) | - (32.8) | - (16.2) | - (45.0) |
| Human | 92.9 | 91.3 | 91.7 | 97.6 | 90.9 |

Table 4 Accuracy on MVP and MVP-mini in parentheses. VideoLLM-performance is slightly greater than random chance, while humans achieve greater than $90 \%$ accuracy on all categories. Results for closed-source models are only shown on MVP-mini due to API costs. Performance is measured via Minimal Pair Score, wherein a model obtains a score iff the prediction for both QA examples of the pair is correct. ${ }^{\dagger}=$ if temperature of LLM is zero.
et al., 2024; Weihs et al., 2022; Du et al., 2023), as this involves reasoning about e.g. object permanence, gravity and trajectories.

VideoLLMs performance on dataset sub-tasks. Some sources in MVP are further divided into more finegrained splits, where each split tests for a specific ability (e.g., object permanence, shape consistency, motion consistency, etc.). In this section we summarize more detailed observations we gathered on these splits.

While performance on all intuitive physics tasks is close to 0\%, we find that LongVU (Qwen2) obtains non-trivial performance on three splits: Gravity-Continuity (39.1\%) and Unchangeableness ( $42.2 \%$ ); with Tarsier-34B performing well on Gravity-Support (35.2\%). Even some of the weaker models can achieve performance clearly above random on our Fine-grained human-object interactions category when looking closer into subsets such as Counterfactual (e.g., Qwen2-VL: 46.5\% LongVU (Qwen2): 43.9\%) and Memory (e.g., LongVU (Qwen2): 40.4\%) examples.

Importance of Data Curation. In Table 5, we explore the effects of the minimal-change pair mining and single-frame bias filtering on model performance. For this exploration we use the smaller MVP-mini (see Appendix A) and report the average performance of five VideoLLMs ${ }^{2}$.

When pairing videos randomly instead of using minimalchange pairs, the average accuracy across tasks is at

[^1]$45.4 \%$, far superior to random chance. Using minimalchange pairs, the average VideoLLM performance significantly drops to $27.3 \%$. This result shows the importance of the minimal-pair framework and suggests that VideoLLMs can frequently leverage shortcut solutions or spurious features to solve QA tasks. Additionally, the average VideoLLM performance drops again by another $2.2 \%$ to $25.1 \%$ by removing single-frame solvable videos, with much larger drops on certain subsets. Note that while Fine-grained robot-object interactions and the Intuitive physics and collisions categories contain almost no single-frame biases, we can see significant drops of $3.5 \%$ and $3.3 \%$ for the other two categories (Fine-grained human-object interactions and Coarse-grained temporal reasoning) with this additional filtering step. Overall, Table 5 confirms that the minimal-change pair mining and single-frame filtering pipeline is effective at mitigating potential shortcut solutions in MVP.

## 5 Related Work

Language biases in Vision-Language models. Vision-andlanguage benchmarks, such as Visual Question Answering (VQA) (Antol et al., 2015; Goyal et al., 2017; Marino et al., 2019) have been found to be vulnerable to language biases as evidenced by the performance of "blind" language-only models. Blind models are routinely shown to be efficient at solving many of the vision-and-language tasks (Goyal et al., 2017; Zeng et al., 2023; Chen et al., 2024a), and can also solve several image-text retrieval

| Model | Overall | Fine-grained human-object interactions | Fine-grained robot-object interactions | Intuitive physics and collisions | Coarse-grained Temporal reasoning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pairing of random videos (with same question) |  |  |  |  |  |
| Avg. VideoLLM Acc. | 45.4 | 36.8 | 40.9 | 19.7 | 84.3 |
| + Pairing of minimally different videos |  |  |  |  |  |
| Avg. VideoLLM Acc. | $\downarrow 18.1$ | 28.7 | $\downarrow 22.3$ | 16.7 | 45.1 |
| + Remove single-frame-solvable examples = final version of MVP |  |  |  |  |  |
| Avg. VideoLLM Acc. | 25.1 $\downarrow 2.2$ | 25.2 $\downarrow 3.5$ | 18.3 $\downarrow 0.3$ | 15.2 $\downarrow 1.5$ | 41.8 $\downarrow 3.3$ |
|  |  |  |  |  |  |

Table 5 We ablate the effect of our main curation steps. Both the automatic pairing of minimal pairs and the single-frame-bias filtering lead to lower average model performance, with an especially large drop once we introduce the minimal pair setup.
benchmarks (Yuksekgonul et al., 2022; Hsieh et al., 2024) using language biases (Lin et al., 2023). Visual Question Answering in the video-language domain (Video-QA) (Li et al., 2024b; He et al., 2024; Xiao et al., 2021; Lei et al., 2018; Majumdar et al., 2024; Tapaswi et al., 2016; Rawal et al., 2024) also exhibits language biases, as shown in the performance of strong language-only baselines (Zhang et al., 2023a; Cores et al., 2024).

Vision-centric biases in Vision-Language models. State of the art vision-language models are shown to be surprisingly unaware of the vision inputs, where they often struggle with simple questions due to incorrect visual grounding (Tong et al., 2024), despite leveraging sufficiently powerful visual embeddings. VLMs are shown to be imprecise at spatial information understanding and geometry (Rahmanzadehgervi et al., 2024; Kamath et al., 2023). Similar biases exists in video-and-language tasks, where VideoLLMs typically exhibit single-frame bias (Buch et al., 2022; Lei et al., 2023) or spatial bias (Cores et al., 2024), where either a single frame is enough to solve the task, or the ordering of the frames is not important. To overcome this bias, benchmarks propose computing temporal certificate sets (Mangalam et al., 2024), key-frame bias (Buch et al., 2022), or investigate temporal understanding through shuffled frame inputs (Cores et al., 2024). In MVP, we operationalize a looser definition of temporal understanding for our filtering pipeline (Section 3) in that we keep an example if it is only solvable given the right key-frame, but discard it if it can be solved with any randomly sampled frame - the intuition being that key-frame identification can already involve temporal reasoning.

Benchmarks addressing vision-and-language biases. Several approaches are proposed in the literature to reduce the aforementioned biases in Vision-Language systems. One promising approach is to use minimally different pairs of inputs (Thrush et al., 2022; Yuksekgonul et al., 2022; Hsieh et al., 2024; Krojer et al., 2022; Wang et al., 2023a), also known as Contrast Sets Gardner et al. (2020), which stem from related work in natural language processing (Levesque et al., 2012; Sakaguchi et al., 2021; McCoy et al., 2019). Minimally different input
pairs restrict the models' abilities to use these biases, as both samples in the pair must be answered correctly to achieve a non-zero score. Similar to MVP, some highly adopted examples of such image-language benchmarks build on top of existing image sources (ARO (Yuksekgonul et al., 2022)), or fix them explicitly (SugarCREPE (Hsieh et al., 2024)). Commonly, the focus is on textual minimal-change pairs, e.g., providing several answer candidates for a question with only slight variations in word order (Yuksekgonul et al., 2022; Cores et al., 2024; Park et al., 2022; Li et al., 2023; Cai et al., 2024). However, textual minimal-change pairs can be susceptible to the same language biases (Hsieh et al., 2024; Wu et al., 2023). Other works, such as in Video-QA, focuses on visual minimal-change pairs. TempCompass (Liu et al., 2024) creates a small set of less than 0.5 K artificial minimally different videos by manipulating the original video, e.g., playing the video in reverse, at a faster speed, or playing one video above the other. Vinoground (Zhang et al., 2024a) scrapes 0.5 K minimally different video pairs from YouTube with the majority following the same pattern: event $A$ before $B$ vs. event $B$ before $A$. Our work differs in several aspects from these (summarized Table 2), notably as well in terms of the scale of curation by showing that minimal video pairs can be procedurally extracted from existing video sources. While our Minimal Pair Score is inspired by Winoground (Thrush et al., 2022), unlike Vinoground, we intentionally do not adopt the Winoground metric directly since we want MVP to be agnostic to whether models can process several videos in one forward pass.

The language biases in existing vision-language benchmarks often stem from the over-reliance on world knowledge and plausible co-occurrences (Hsieh et al., 2024; Goyal et al., 2017). Thus, MVP focuses on short videos with "basic" perceptual skills (spatial, temporal, or intuitive physics), which requires understanding of physical world properties (Yi* et al., 2020; Chen et al., 2022; Jassim et al., 2024; Riochet et al., 2022; Bear et al., 2021; Margoni et al., 2024; Baillargeon et al., 1985), reducing the space for blind LLMs to rely on their cultural knowledge.

## 6 Discussion and Limitations

Going back to our initial question, our results suggest that VideoLLMs do not yet perceive and understand the world as reliably as humans. After evaluating various state-of-the-art VideoLLM models for physical world understanding on MVP, the best model obtains only $40.2 \%$ average accuracy, while human performance is $92.9 \%$. Yet, VideoLLMs are not completely blind. On some sub-categories of spatio-temporal understanding and intuitive physics, VideoLLMs can perform significantly better than random chance. Overall, our empirical evaluation shows that current VideoLLMs are still far from matching human performances on all tested tasks, calling for more research in this direction to develop better training data for world modelling, as well as novel learning criteria and model architectures. We anticipate MVP to help the development of the next generation of visual systems to perceive the world as robustly as humans.

Limitations: No benchmark comes without limitations. First, it is possible that forcing the model to output a single letter without any room for free-form reasoning (CoT) limits its performance. Additionally, using an automated curation approach will not be able to fully remove noisy examples; through manual inspection, we found some of the examples to be too simple, and a few others to be ambiguous, although we note that these noisy samples only represent a small subset of the overall data.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716-23736, 2022.
Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425-2433, 2015.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren

Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Renee Baillargeon, Elizabeth S Spelke, and Stanley Wasserman. Object permanence in five-month-old infants. Cognition, 20(3):191-208, 1985.

Daniel Bear, Elias Wang, Damian Mrowca, Felix Binder, Hsiao-Yu Tung, Pramod RT, Cameron Holdaway, Sirui Tao, Kevin Smith, Fan-Yun Sun, Fei-Fei Li, Nancy Kanwisher, Josh Tenenbaum, Dan Yamins, and Judith Fan. Physion: Evaluating physical prediction from vision in humans and machines. In J. Vanschoren and S. Yeung, editors, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. https://datasets-benchmarks-proceedings. neurips.cc/paper_files/paper/2021/file/ d09bf41544a3365a46c9077ebb5e35c3-Paper-round1.pdf.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. A large annotated corpus for learning natural language inference. In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632-642, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. https://aclanthology. org/D15-1075.

Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the" video" in video-language understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2917-2927, 2022.

Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? arXiv [cs.CV], March 2024a. http://arxiv.org/abs/2403.20330.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

Zhenfang Chen, Kexin Yi, Yunzhu Li, Mingyu Ding, Antonio Torralba, Joshua B Tenenbaum, and Chuang Gan. Comphy: Compositional physical reasoning of objects and events from videos. In International Conference on Learning Representations, 2022.

Daniel Cores, Michael Dorkenwald, Manuel Mucientes, Cees GM Snoek, and Yuki M Asano. Tvbench: Redesigning video-language evaluation. arXiv preprint arXiv:2410.07752, 2024.

Ido Dagan, Dan Roth, Fabio Zanzotto, and Mark Sammons.

Recognizing textual entailment: Models and applications. Morgan \& Claypool Publishers, 2013.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-theart multimodal models. arXiv preprint arXiv:2409.17146, 2024.

Danny Driess, F. Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Ho Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Peter R. Florence. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, 2023. https://api.semanticscholar. org/CorpusID:257364842.

Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B. Tenenbaum, Leslie Kaelbling, Andy Zeng, and Jonathan Tompson. Video language planning, 2023. https://arxiv.org/abs/2310.10625.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
J. Gao, Chen Sun, Zhenheng Yang, and Ramakant Nevatia. Tall: Temporal activity localization via language query. In $I C C V, 2017$.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, Sameer Singh, Noah A. Smith, Sanjay Subramanian, Reut Tsarfaty, Eric Wallace, Ally Zhang, and Ben Zhou. Evaluating models' local decision boundaries via contrast sets. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1307-1323, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.117. https://aclanthology.org/2020.findings-emnlp.117.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904-6913, 2017.

Xuehai He, Weixi Feng, Kaizhi Zheng, Yujie Lu, Wanrong Zhu, Jiachen Li, Yue Fan, Jianfeng Wang, Linjie Li, Zhengyuan Yang, et al. Mmworld: Towards multidiscipline multi-faceted world model evaluation in videos. arXiv preprint arXiv:2406.08407, 2024.

Matthew Honnibal and Ines Montani. spaCy 2: Natural
language understanding with Bloom embeddings, convolutional neural networks and incremental parsing. To appear, 2017.

Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in neural information processing systems, 36, 2024.

Serwan Jassim, Mario Holubar, Annika Richter, Cornelius Wolff, Xenia Ohmer, and Elia Bruni. Grasp: A novel benchmark for evaluating language grounding and situated physics understanding in multimodal language models. In Kate Larson, editor, Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 6297-6305. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi: 10.24963/ijcai.2024/696. https://doi.org/10.24963/ijcai. 2024/696. Main Track.

Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. In Fortieth International Conference on Machine Learning, 2023.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901-2910, 2017.

Amita Kamath, Jack Hessel, and Kai-Wei Chang. What's "up" with vision-language models? investigating their struggle with spatial reasoning. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9161-9175, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.568. https://aclanthology. org/2023.emnlp-main.568.

Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-andlanguage navigation in continuous environments. In $E C C V$, 2020.

Benno Krojer, Vaibhav Adlakha, Vibhav Vineet, Yash Goyal, Edoardo Ponti, and Siva Reddy. Image retrieval from contextual descriptions. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3426-3440, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.241. https: //aclanthology.org/2022.acl-long. 241.

Hugo Laurençon, Andrés Marafioti, Victor Sanh, and Léo Tronchon. Building and better understanding visionlanguage models: insights and future directions. arXiv preprint arXiv:2408.12637, 2024.

Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. TVQA: Localized, compositional video question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and


[^0]:    ${ }^{1}$ We release MVP-mini for fast eval and lower costs of API models.

[^1]:    ${ }^{2}$ LLaVA-OV, VideoChat, Qwen2-VL, LongVU (Qwen2), Tarsier-7B

