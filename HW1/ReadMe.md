
1. Use python3
2. Use liac-arff -> pip3 install liac-arff
3. Usage:
 ```
  $ python3 main.py <path to training.arff> <path to test.arff>  <optional confidence level>
  when confidence level is not given, All of them [99, 95, 80, 50, 0] will be used one by one.
 ```
4. Example Output (using DTree.py in main):
 ```
    $ python3 main.py ../DataSets/training_subsetD.arff ../DataSets/testingD.arff 
    Building and testing trees for Confidences:  [99, 95, 80, 50, 0]
    Loading Train File: ../DataSets/training_subsetD.arff
    Training Loaded: 14.503 secs!

    Loading Test File: ../DataSets/testingD.arff
    Test Loaded: 9.147secs!
    Target Attribute 274 -> ('Class', ['True', 'False']) 

    Growing Tree for Confidence: 99.00 %
    Tree is ready! (13.73 secs)
    Starting Tree analyze:
    Analysis complete (0.00 secs) 
	    Nodes: 8 
	    Leaves: 20 
	    SneakPeak: [13, 8, 167, 166]
    Starting Train Classification   
    Training classification complete (0.49 secs)
	    Accuracy: 80.59 % 
	    Precision: 81 % 
	    Recall: 0.74 %
    Positive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Negetive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Starting Test classification
    Training classification complete (0.43 secs)
	    Accuracy: 74.88 % 
	    Precision: 87 % 
	    Recall: 0.22 %
    Positive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Negetive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse')]

    Growing Tree for Confidence: 95.00 %
    Tree is ready! (14.05 secs)
    Starting Tree analyze:
    Analysis complete (0.00 secs) 
    	Nodes: 9 
	    Leaves: 21 
	    SneakPeak: [13, 8, 167, 166]
    Starting Train Classification
    Training classification complete (0.49 secs)
	    Accuracy: 80.60 % 
	    Precision: 82 % 
	    Recall: 0.74 %
    Positive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Negetive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Starting Test classification
    Testing classification complete (0.42 secs)
	    Accuracy: 74.88 % 
	    Precision: 87 % 
	    Recall: 0.22 %
    Positive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse')]
    Negetive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse')]

    Growing Tree for Confidence: 80.00 %
    Tree is ready! (116.36 secs)
    Starting Tree analyze:
    Analysis complete (0.00 secs) 
    	Nodes: 152 
    	Leaves: 280 
    	SneakPeak: [13, 8, 207, 2]
    Starting Train Classification
    Training classification complete (1.15 secs)
    	Accuracy: 81.33 % 
	    Precision: 98 % 
	    Recall: 4.41 %
    Positive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Starting Test classification
    Testing classification complete (1.26 secs)
	    Accuracy: 75.03 % 
	    Precision: 54 % 
	    Recall: 4.39 %
    Positive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]

    Growing Tree for Confidence: 50.00 %
    Tree is ready! (173.14 secs)
    Starting Tree analyze:
    Analysis complete (0.02 secs) 
    	Nodes: 1118 
    	Leaves: 2362 
    	SneakPeak: [13, 8, 207, 2]
    Starting Train Classification
    Training classification complete (2.08 secs)
    	Accuracy: 85.21 % 
    	Precision: 98 % 
    	Recall: 24.47 %
    Positive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Starting Test classification
    Testing classification complete (1.62 secs)
    	Accuracy: 73.95 % 
    	Precision: 44 % 
    	Recall: 12.91 %
    Positive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]

    Growing Tree for Confidence: 0.00 %
    Tree is ready! (301.10 secs)
    Starting Tree analyze:
    Analysis complete (0.27 secs) 
    	Nodes: 14349 
    	Leaves: 23939 
    	SneakPeak: [13, 8, 207, 2]
    Starting Train Classification
    Training classification complete (3.62 secs)
    	Accuracy: 93.05 % 
    	Precision: 90 % 
    	Recall: 71.72 %
    Positive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Train: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Starting Test classification
    Testing classification complete (2.31 secs)
    	Accuracy: 67.08 % 
    	Precision: 32 % 
    	Recall: 29.45 %
    Positive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Negetive path in Test: [(13, 'Friend/Co-worker'), (8, 'spouse'), (207, "\\'(-inf-0.5]\\"), (2, 'Male'), (104, "\\'(-inf-0.5]\\"), (18, "\\'(-inf-0.5]\\")]
    Total execution time: 656.55 secs
 ```