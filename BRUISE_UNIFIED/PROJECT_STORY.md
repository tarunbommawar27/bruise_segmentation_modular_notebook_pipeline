# The Bruise Project, told as a story

*A plain-language account of everything we built, why we built it that way, what
happened, and what it means. No maths, no jargon left unexplained.*

*Companion to `PROJECT_HANDBOOK.md`, which is the technical version. Where they
disagree, the handbook is right — it is generated from the code.*

**Last updated: 7 August 2026.**

---

## Chapter 1 — The problem

A bruise is evidence. In cases of suspected assault or abuse, a photograph of a
bruise can matter in court. But bruises are hard to see — especially on darker
skin, especially when they are small, especially when the lighting is poor.

So the question we set out to answer is:

> **Can a computer look at a photograph of skin and outline the bruise?**

Not "is there a bruise somewhere in this picture" — actually draw the shape, so
you can measure it, track it over days, and compare it consistently between
examiners.

### The one thing that shaped everything else

Early on, we realised the clinical question is not *"how neatly did the computer
outline the bruise?"* It is:

> **"Did it find the bruise at all?"**

An outline that is 10% too wide is a minor annoyance. An outline that doesn't
exist — the computer looked at a photograph containing a bruise and reported
nothing — is a failure that matters. We call that a **complete miss**, and it has
been the headline measure of this project ever since.

Hold onto that. It comes back in Chapter 10 and changes the whole story.

---

## Chapter 2 — The data

**1,016 photographs from 143 people.**

Each photograph comes with a hand-drawn outline of the bruise, made by a trained
human labeller. That outline is the "right answer" the computer is trying to
reproduce.

We split them into three piles:

| pile | photos | what it's for |
|---|---|---|
| **training** | 697 | the computer learns from these |
| **validation** | 134 | we check progress on these while training |
| **test** | 185 (from 28 people) | untouched until the very end |

### The rule we never broke: split by person, not by photo

If someone has six photographs of the same bruise taken on different days, all
six go into the same pile. Never split a person across piles.

Why this matters enormously: if a photo of your arm on Monday were in the
training pile and a photo of the *same arm* on Tuesday were in the test pile, the
computer would score brilliantly — by recognising your arm, not by understanding
bruises. It would look like a triumph and be worthless in the field.

This is the single most common way medical AI results turn out to be fake, and
guarding against it is why our numbers look modest compared to some published
work.

---

## Chapter 3 — What a model actually is, in plain terms

Every model in this project has two halves.

**The encoder — the eyes.** It looks at the photograph and progressively boils it
down. It starts with raw pixels and ends with a small grid of rich descriptions:
*"there's a reddish-purple patch with soft edges here"*. It throws away detail to
gain understanding.

**The decoder — the hand.** It takes those descriptions and draws the outline
back at full size, deciding for every pixel: bruise, or not bruise.

An analogy: the encoder is a doctor glancing at a photo and thinking "bruise,
upper arm, medium sized, fading". The decoder is the doctor then taking a pen and
tracing its exact border.

### Some specifics, since you asked

**Images are 640 × 640 pixels.** Every model sees exactly the same size. Bigger
would be better for tiny bruises but costs memory and time; 640 was the largest
we could run every model at fairly.

**Input: 3 channels.** Red, green, blue — an ordinary colour photo.

**Output: 1 channel.** A single number per pixel: *how confident am I that this
pixel is bruise?* Not "which of ten things is this" — just one yes/no question,
409,600 times per photo. That's why it's called **binary** segmentation.

---

## Chapter 4 — Teaching the model: the loss function

A **loss function** is the score that tells the model how wrong it is. Training
is nothing more than: guess, get a score, adjust, repeat — about a hundred times
through all 697 training photos.

We used **two scores added together**, and the reason is worth understanding.

**Score 1 — pixel-by-pixel correctness (BCE).** For each pixel, how confidently
wrong were you? Harsh on confident mistakes, gentle on hesitant ones.

Its weakness: a bruise typically covers **2–3% of the photograph**. A lazy model
could answer "no bruise" for every single pixel and be 97% correct. This score
alone would let it get away with that.

**Score 2 — overlap (Dice).** How much does your outline overlap the true
outline, as a fraction of their combined size? A model that answers "nothing
anywhere" scores **zero** here. No hiding.

**Together they cover each other's blind spots.** Overlap forces the model to
find the bruise; pixel-correctness gives smooth, stable feedback when overlap is
still near zero and would otherwise give almost no signal to learn from.

**One extra detail:** for the SegFormer family we also scored an intermediate
layer at 40% weight (`aux_weight = 0.4`). Think of it as checking the student's
working, not just the final answer — it helps the earlier layers learn faster.
Other architectures don't have a place to attach it, so they got 0.

---

## Chapter 5 — The recipe, held fixed on purpose

Every model trained under **identical conditions**:

- up to **100 passes** through the data, stopping early if it stopped improving
  for 15 passes in a row
- **AdamW** — the standard method for nudging the model in the right direction
- the eyes learn slowly (0.00006), the hand learns **10× faster** (0.0006)
- **three separate runs** of everything with different random starting points
  (seeds 0, 1, 2)

### Why the eyes learn slower than the hand

Most encoders arrive **pre-trained** — they've already seen millions of ordinary
photographs and know what edges, textures and colours look like. That knowledge
is valuable and fragile, so we nudge it gently. The decoder starts from nothing
and needs to catch up fast.

### Why three runs of everything

Train the same model twice with different random starts and you get slightly
different results. If we ran each model once, we'd be reporting luck. Three runs
tells us how much of any difference is real.

### Why we refused to tune each model separately

This is the discipline that makes the whole study mean something. If we'd tuned
each architecture lovingly, the winner would be *"the one we spent the most time
on"*, not *"the best architecture"*. Same recipe for everyone, even where a
model's own paper recommends something different.

**The honest cost:** Fast-SCNN's own paper uses settings about 100× different
from ours. It came last. Some of that is probably our recipe, not the
architecture, and we say so in the limitations rather than quietly re-tuning one
contestant.

---

## Chapter 6 — Choosing the cut-off

The model doesn't output "bruise" or "not bruise". It outputs a confidence for
every pixel. Somebody has to decide: **above what confidence do we call it a
bruise?**

We tested 481 different cut-offs on the validation photos.

**What we found was surprising:** it barely mattered. For our main model, moving
the cut-off across a huge range changed the overlap score by **0.009** — nine
thousandths. That's not a peak with a clear best answer. That's a flat plateau
with noise on it.

**So we did something unusual.** Instead of picking the highest point — which
would be fitting to random noise — we treated every cut-off that was
statistically tied for first as equally good, and broke the tie by asking: *which
one misses the fewest bruises entirely?*

Cut-offs that are equally good at outlining are **not** equally good at finding.
That choice is invisible in the headline score and directly serves the thing we
actually care about.

---

## Chapter 7 — The contestants

We trained a lot of architectures. Here's the cast, in plain terms.

### The accurate ones

**SegFormer** (three sizes: B0 small, B2 medium, B5 large). A modern
transformer-based design. Its trick is looking at the image at four different
zoom levels at once and combining them — which turns out to matter a great deal
for small bruises.

**U-Net.** The classic medical-imaging workhorse, invented in 2015 and still
everywhere. Shaped like a U: compress down, expand back up, with shortcuts
connecting matching levels so fine detail isn't lost.

**DeepLabV3+.** Another established design, built to see objects at multiple
scales using cleverly spaced-out filters.

### The fast ones (for phones and tablets)

**Fast-SCNN, LR-ASPP MobileNetV3, TopFormer, PP-MobileSeg.** All built for speed
on weak hardware. The question they answer: *how much accuracy do you give up to
run on a phone?*

**YOLO.** Famous for real-time object detection; we adapted it to draw outlines.

### The results

| model | size | overall score | **complete misses** (out of 185) |
|---|---|---|---|
| SegFormer-B2 | 27.4 M | 0.769 | **0** |
| SegFormer-B0 (taught) | 3.7 M | 0.768 | **0** |
| SegFormer-B0 (direct) | 3.7 M | 0.766 | **1** |
| DeepLabV3+ | 40 M | 0.758 | 5 |
| U-Net | 32 M | 0.757 | 7 |
| LR-ASPP MobileNet | 3.2 M | 0.698 | 2 |
| YOLO | 1.6 M | 0.702 | 12 |
| Fast-SCNN | 1.1 M | 0.605 | 13 |

Speed, measured on the same hardware: Fast-SCNN 282 frames/second, LR-ASPP 204,
TopFormer 164, PP-MobileSeg 96.

**Read the two right-hand columns against each other.** U-Net has a *better*
overall score than LR-ASPP MobileNet — and misses **three times as many bruises
entirely**. The headline number and the thing we care about disagree. That
disagreement becomes the story in Chapter 10.

---

## Chapter 8 — The bug that taught us the most

Worth telling because it explains a rule we now apply everywhere.

Photographs arrive as numbers from 0 to 255 per colour. Before a model sees them,
they get rescaled — and **different models expect different rescalings**.
SegFormer wants one convention. YOLO wants another, because its internal
machinery has fixed expectations baked in from its original training.

We fed YOLO the wrong convention.

It didn't crash. It didn't warn. It trained happily to completion and produced a
plausible-looking score of **0.479** — clearly worse than everything else, and we
nearly wrote it up as "YOLO is not suited to this task."

It was suited. It was being handed miscalibrated photographs. No cut-off could
recover it, because the damage was upstream of everything.

**The lesson, now a project rule:**

> **A number that appears without an error message is not a number that was
> produced correctly.**

We hit the same class of bug again in Chapter 12, and the guard we'd built from
this one is what caught it.

---

## Chapter 9 — Teaching small models to imitate big ones

**Knowledge distillation** — a big accurate model (the *teacher*) trains a small
fast model (the *student*).

The insight: the teacher's hesitations are informative. If it says "85% sure this
is bruise" rather than a flat yes, the student learns *where the genuinely hard
edges are* — information that the hand-drawn outline, being a hard yes/no, simply
doesn't contain.

We tried a lot of ways to do this.

| technique | the idea in one line |
|---|---|
| **Response** | copy the teacher's final answer |
| **CWD** | match the teacher's internal patterns channel by channel |
| **BPKD** | pay extra attention to boundaries, where the disagreement is |
| **Ensemble** | average several teachers first, then imitate the average |
| **Adaptive** | trust the teacher more where it's confident |
| **Boundary / Hard / Group / Full** | four variants weighting edges, difficult cases, skin-tone groups |
| **Angular** | match the *directions* of internal patterns, not their sizes |
| **Reliability-gated** | only listen to the teacher on images where it has proven trustworthy |
| **Multi-teacher routing** | pick the best teacher *per image* |

Ten arms in the main sweep, plus reliability gating across five student
architectures, plus multi-teacher.

### What happened

**Almost nothing.** Every arm landed within a whisker of the plainly-trained
model. Not worse — genuinely comparable — but not better either.

The most instructive failure was **multi-teacher routing**. We first checked
whether it *could* work: if a perfect oracle picked the ideal teacher for every
image, would that be better? Yes — meaningfully so. Real headroom existed.

Then we built the real thing. The student captured **none** of it.

The reason turned out to be sharp: when we blended the teachers' opinions
together, the blend scored **0.723** — *worse than every single teacher in the
pool*, all of whom were above 0.77. Averaging three good but differently-shaped
outlines produces a blurry consensus that is worse than any of them.

**What this taught us:** the bottleneck was never teacher quality. It was the
transfer. Adding a bigger teacher to a pipeline that degrades its inputs changes
the wrong variable — which is why we stopped adding distillation arms.

### The one thing distillation did buy

The taught 3.7 M model reached **zero complete misses**, matching a teacher seven
times its size. We can't prove statistically that teaching *caused* it — the
comparison comes back "no measurable difference" — but a phone-sized model
matching a server-sized one is the practical result the project needed.

---

## Chapter 10 — The finding that reframed everything

Partway through, we asked a question nobody had asked: **how much do the human
labellers agree with each other?**

Three people had independently outlined the same 185 test photographs. We
compared them.

| comparison | agreement |
|---|---|
| labeller B vs labeller C | **0.755** |
| labeller A vs labeller B | **0.581** |
| labeller A vs labeller C | **0.581** |
| *our best model vs the consensus* | *0.769* |

Read that again.

**Our models agree with the consensus better than the humans agree with each
other.**

### What this means

There is a ceiling, and we are at it. When two experts hand-drawing the same
bruise differ by 0.17, a difference of 0.005 between two models is not a real
difference — it is inside the noise of what "correct" even means.

We confirmed it statistically: across the seven headline models, a formal test
for *"is any of these different from any other?"* returns **no**. p = 0.61.
Nothing to see.

### Why this is good news, not bad

It kills a whole category of false claims. Nobody on this project can now write
*"our method improves Dice by 0.008"* and have it mean anything. That is a real
protection.

And it forces the right question: **if overall accuracy can't separate the
models, what can?**

---

## Chapter 11 — Where the failures actually live

So we went back to the complete misses and asked where they happen.

We sorted all 185 test photographs by how big the bruise was and split them into
ten equal groups — group 1 the smallest bruises, group 10 the largest. Then we
counted, across all 39 model versions, where the complete misses fell.

```
smallest  group 1   58 misses  ██████████████████████████████
          group 2    8         ████
          group 3   21         ███████████
          group 4   19         ██████████
          groups 5-10  13 combined  ██████
```

**Nine out of every ten failures happen on the smallest 40% of bruises. Half of
them happen on the smallest 10%.**

And the models separate far more clearly there:

| measure | spread across all models |
|---|---|
| overall score | 0.605 → 0.775 (a range of 0.17) |
| finding small bruises | 0.341 → 0.844 (a range of **0.50**) |

**Three times the separation.** The models are not equally good. We had just been
measuring in the wrong place — averaging over easy photographs where everyone
does fine.

### A second discovery in the same pass

There are two ways to fail, and we had been counting them as one.

- **The model outputs nothing.** It doesn't know, and says so.
- **The model outputs a large region in completely the wrong place.** Confident,
  and wrong.

For a clinical tool these are not remotely equivalent. One is a system saying
"please look yourself". The other is a system pointing at the wrong part of the
body.

Fast-SCNN has 13 complete misses — **5 of them are confident errors**. One of its
distilled variants has 6 misses of which only 1 is an honest "I don't know".
Meanwhile U-Net's 7 misses are **all** honest, and YOLO's 12 are **all** honest.

Nobody had separated these before. It changes how you'd deploy any of them.

---

## Chapter 12 — Do foundation models help?

"Foundation models" are enormous models trained on vast image collections, meant
to be adapted to anything. Some are trained specifically on medical images. The
obvious question: **does training on medical images help with bruises?**

We designed a cheap test. Freeze three different sets of "eyes" — don't let them
learn anything new — and attach the simplest possible "hand". Whatever score
comes out reflects purely what the eyes already knew.

Three contestants:

- **MedSigLIP** — 428 million parameters, trained on medical images including skin
- **DINOv2** — 87 million, trained on ordinary internet photographs, **zero
  medical data**
- **ResNet-50** — 24 million, the standard baseline

DINOv2 was included as a check: if a model with *no* medical training matches the
medical one, then any advantage isn't about medicine.

### The result, and the near-miss

| eyes | score |
|---|---|
| **DINOv2** (no medical data) | **0.657** |
| MedSigLIP (medical) | 0.491 |
| ResNet-50 (baseline) | 0.123 |

The model with **no medical training beat the medical one**, clearly.

But here's the part worth telling. **The first time we ran this, we got the
opposite answer** — a confident result saying medical training explained 86% of
the gain.

It was wrong. A loading function had silently built a *randomly initialised*
model instead of DINOv2 and returned it without complaining. We were comparing
MedSigLIP against noise. The tell was a parameter count 8% off from published —
caught only because Chapter 8's lesson had made us add that check.

**The arm we added specifically to stop us publishing a wrong claim was itself
the broken one.** If we'd skipped DINOv2, or left it broken, we would have
published "medical pretraining helps" with confidence.

### And we still won't quote the result

Because there's a confound. The three contestants see the image at different
levels of coarseness, and the scores line up **exactly** with the coarseness
ordering. A model that sees finer detail can draw a better boundary regardless of
what it understands.

A twenty-minute control run settles it. **It hasn't run yet**, so this result
stays unpublished. That's not indecision — it's the difference between a
defensible claim and a retraction.

---

## Chapter 13 — Fairness

A serious concern for this work: does the system work worse on darker skin?

We measured, across every model and every skin-tone group. **Twenty of twenty-one
tests found no significant difference.**

The one that did had its worst group at *tan* skin, not dark — which doesn't fit
the expected pattern. And where our YOLO variant showed a gap, the worst group
was **light** skin.

### The catch — and we have now checked it

Bruise size and skin tone are tangled in our dataset. And Chapter 11 established
that **size is the strongest predictor of failure there is.** So an apparent
skin-tone effect might just be a size effect wearing a skin-tone label.

**We measured the tangle.** It's real and large:

| skin-tone group | typical bruise size | share that are small bruises |
|---|---|---|
| Light | 8,085 px | **59%** |
| Dark | 13,751 px | 33% |

Light-skin photographs are **almost twice as likely to be small-bruise
photographs**, with bruises 40% smaller on average. So every skin-tone comparison
we had published was measuring two things at once — and now we know by how much.

### Then we redid the comparison holding size fixed. Three surprises.

**1. The gaps didn't shrink. They got bigger** — in four of our five models. We
expected the size tangle to explain the gaps. It doesn't.

**2. Darker skin is never the worst group** — in any model — and in two models
it's the **best**. Whatever differences exist here, they do not run light-to-dark.

**3. The one clear, repeatable signal runs the opposite way.** Our two weakest
models (YOLO and U-Net) do worst on **light** skin, by a wide margin, and this
survives the size correction. We had previously written that gap off as probably
a size artifact. It isn't.

### Why we still aren't making a fairness claim

Because of who's in the data. Two skin-tone groups — Brown and Tan — have only
**three and four people** in the relevant comparison. You cannot draw a conclusion
about a group from three people, and our analysis refuses to try: **10 of 25 cells
come back with no answer at all.**

For our three best models, the "worst group" is always one of those two — so their
gaps rest entirely on numbers nobody can verify.

**So the honest statement is not "the models are fair" and not "the models are
unfair". It is: this dataset cannot answer the question for those groups.**

That's the most useful thing this analysis produced, and it's a shopping list, not
a conclusion: **we need roughly twice as many Brown and Tan participants before
the fairness question can be answered at all.** That goes to whoever plans the
next round of data collection.

We also tried *building* fairness in, by weighting training toward
under-performing skin-tone groups. It produced one of the **worst** results in
its group. We're not pursuing that direction until the measurement question is
settled — mechanism before measurement is how projects fool themselves.

---

## Chapter 14 — Where we are today

### What we know

1. **Every model is at the human-agreement ceiling.** Overall accuracy cannot
   separate them, and nobody should claim otherwise.
2. **Complete misses are a small-bruise problem.** 89% of all failures are on the
   smallest 40% of bruises.
3. **Small-bruise performance separates models three times better than overall
   accuracy does** — and early statistics say those differences are real.
4. **The best-looking model is not the best model.** U-Net has the study's best
   typical score and worse small-bruise miss containment than a model nine times
   smaller.
5. **Medical-specific pretraining did not beat generic pretraining** — pending one
   control run.
6. **Two kinds of failure exist** — honest silence and confident error — and they
   should never be counted together again.

### Our pick

**SegFormer-B0, trained directly.** 3.7 million parameters, small enough for a
phone, zero-to-one complete misses, statistically indistinguishable from models
23× its size, and it needs no teacher to produce.

*One caveat we owe:* the SegFormer family carries a non-commercial licence. If
this ever ships commercially, the whole podium is unavailable and the best clean
alternative is measurably worse. Worth knowing which constraint applies before
that decision is forced.

### A correction, on the record

While preparing this, we found that a claim we'd made the day before — that a
small mobile model beat the largest teacher at finding small bruises — **reversed**
when tested properly. It had come from 19 photographs and a single training run.
Under the proper test on 74 photographs, the big teacher wins clearly.

We're recording it because it's the whole point of building the test. The
analysis that caught it took minutes and needed no GPU.

---

## Chapter 15 — What happens next

Our compute cluster is currently down, so the work is ordered by what doesn't
need it.

**Right now, no cluster required:**

1. **Confirm the small-bruise result.** Already run once on a laptop. It reads
   scores we saved earlier, not the models themselves, which is why no GPU is
   involved.
2. **Redo the fairness analysis accounting for bruise size.** Free, and it either
   honestly closes the fairness question or finds something real underneath the
   confound.
3. **Prepare the second dataset (Fenwick).** Measure its bruise sizes and its
   labeller agreement. If it has no small bruises it can't help with our actual
   problem — a five-minute answer worth having in advance.

**Waiting on the cluster:**

4. **The twenty-minute control run** that decides whether Chapter 12 is
   publishable.
5. **Alternate-light-source photographs.** Bruises can be clearer under special
   lighting. The plan: use those clearer photos to draw *better outlines*, then
   train on ordinary photos with those better outlines. One caution — in the data
   we've inspected, special lighting revealed only **one** bruise out of 137 that
   ordinary light missed. Worth checking on our own data before building anything.
6. **More data.** With only 19 truly-small bruises in our test set, some questions
   simply cannot be answered no matter how cleverly we analyse. More small
   bruises is the one thing that reliably helps.

---

## The short version

We built and compared about forty model variants, and the most valuable things we
produced were not models.

They were: **a ruler that shows the models are all at the limit of what "correct"
means**; **the discovery that failure is concentrated almost entirely on small
bruises**; **the distinction between a system that says "I don't know" and one
that confidently points at the wrong place**; and **three separate occasions where
a careful check stopped us publishing something false.**

The model we'd deploy is small enough for a phone and misses essentially nothing.
The remaining problem is sharply defined: **small bruises**. That's a much better
place to be than a slightly higher average.
