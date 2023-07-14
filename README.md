## Pulmoary Fibrosis
- Pulmonary Fibrosis is a disorder with no known cause and no known cure, created by scarring of the lungs.
- Outcomes of it can range from long-term stability to rapid deterioration.
- Current methods make fibrotic lung diseases difficult to treat, even with access to a chest CT scan. In addition, the wide range of varied prognoses create issues organizing clinical trials. Finally, patients suffer extreme anxiety—in addition to fibrosis-related symptoms—from the disease’s opaque path of progression. 

## Project Goal
- To develop a prediction model to determine the severity of decline in lung function for pateints with fibrotic lung disease. The aim is to provide patients and clinicians with a better understanding of the disease progression and prognosis.

## Intuition
- Large data sets are in fact large collections of small data sets. For example, in areas like personalised medicine and recommendation systems, there might be a large amount of data, but there is still a relatively small amount of data for each patient or client, respectively. To customise predictions for each person it becomes necessary to build a model for
each person—with its inherent uncertainties—and to couple these models together in a hierarchy so that information can be borrowed from other similar people. We call this the personalisation of models, and it is naturally implemented using hierarchical Bayesian approaches. - Ghahramani (2015)

## First look
![CT Scan](https://github.com/parthshah231/pulmonary_fibrosis/blob/master/README/test3.gif)

- The gif above showcases a CT scan, captured while the patient is either breathing in or holding their breath.
- The patient is lying on their back (in a supine position). Each image represents a different cross-section (axial) of the thorax (the area between the neck and belly)

## Bayesian Modeling
What is Bayesian Modeling?
- Bayesian modeling is a statistical approach that combines our previous knowledge and current data to make educated predictions. It allows us to update our initial beliefs based on new evidence, providing a way to handle uncertainty by offering a range of possible outcomes instead of one fixed answer. Essentially, it's like updating our best guess using both what we knew before and what we've just learned.

``