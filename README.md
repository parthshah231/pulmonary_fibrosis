## Pulmoary Fibrosis
- Pulmonary Fibrosis is a disorder with no known cause and no known cure, created by scarring of the lungs. Outcomes of this disease can range from long-term stability to rapid deterioration.
- Current methods make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. In addition, the wide range of varied prognoses create issues organizing clinical trials. Finally, patients suffer extreme anxiety—in addition to fibrosis-related symptoms—from the disease’s opaque path of progression. 

## Project Goal
- To develop a prediction model to determine the severity of decline in lung function for pateints with fibrotic lung disease. The aim is to provide patients and clinicians with a better understanding of the disease progression and prognosis.

## First look
![CT Scan](https://github.com/parthshah231/pulmonary_fibrosis/blob/master/README/ct_scan.png)

- The above plot showcases a CT scan, captured while the patient is either breathing in or holding their breath.
- The patient is lying on their back (in a supine position). Each image represents a different cross-section (axial) of the thorax (the area between the neck and belly)

## Bayesian Modeling
What is Bayesian Modeling?
- Bayesian modeling is a statistical approach that combines our previous knowledge and current data to make educated predictions. It allows us to update our initial beliefs based on new evidence, providing a way to handle uncertainty by offering a range of possible outcomes instead of one fixed answer. Essentially, it's like updating our best guess using both what we knew before and what we've just learned.

## Intuition
The project plans pn utilizing the concept of personalized hierarchical Bayesian models, in line with Ghahramani's suggestion that "Large data sets are in fact large collections of small data sets." This encapsulates the essence of my approach.
In this context, I are developing individualized models for each patient with pulmonary fibrosis, using their unique clinical and demographic data. This model personalization will recognize the wide variance in disease progression and prognosis among patients.