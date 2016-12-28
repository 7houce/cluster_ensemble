1. Title of Database: Robot execution failures
   Note: it includes 5 different datasets; see 4.

2. Sources:
   (a) Creators / donors:
       -- Luis Seabra Lopes and Luis M. Camarinha-Matos
          Universidade Nova de Lisboa, Monte da Caparica, Portugal
   (b) Date received: April 1999 

3. Past Usage:
   (a) Some publications where it was described/used
       -- Seabra Lopes, L. (1997) "Robot Learning at the Task Level:
          a Study in the Assembly Domain", Ph.D. thesis, Universidade
          Nova de Lisboa, Portugal.
       -- Seabra Lopes, L. and L.M. Camarinha-Matos (1998) Feature 
          Transformation Strategies for a Robot Learning Problem, 
          "Feature Extraction, Construction and Selection. A Data Mining 
          Perspective", H. Liu and H. Motoda (edrs.), 
          Kluwer Academic Publishers. 
       -- Camarinha-Matos, L.M., L. Seabra Lopes, and J. Barata (1996) 
          Integration and Learning in Supervision of Flexible Assembly Systems, 
          "IEEE Transactions on Robotics and Automation", 12 (2), 202-219. 
   (b) Indication of what attribute(s) were being predicted 
       -- The class of execution failure; see 9.
   (c) Indication of study's results
       -- Part of the results is concerned with feature transformation; see 4.
       -- Another set of results is concerned with evaluation of
          data mining algorithms.

4. Relevant Information
   -- The donation includes 5 datasets, each of them defining a different
      learning problem:
      - LP1: failures in approach to grasp position
      - LP2: failures in transfer of a part
      - LP3: position of part after a transfer failure
      - LP4: failures in approach to ungrasp position
      - LP5: failures in motion with part
   -- Feature transformation strategies
      In order to improve classification accuracy, a set of five feature
      transformation strategies (based on statistical summary features,
      discrete Fourier transform, etc.) was defined and evaluated.
      This enabled an average improvement of 20% in accuracy. The most 
      accessible reference is [Seabra Lopes and Camarinha-Matos, 1998].

5. Number of instances in each dataset
   -- LP1: 88
   -- LP2: 47
   -- LP3: 47
   -- LP4: 117
   -- LP5: 164

6. Number of features: 90 (in any of the five datasets)

7. Feature information
   -- All features are numeric (continuous, although integers only).
   -- Each feature represents a force or a torque measured after
      failure detection; each failure instance is characterized in terms
      of 15 force/torque samples collected at regular time intervals
      starting immediately after failure detection; 
      The total observation window for each failure instance was of 315 ms.
   -- Each example is described as follows:

                 class
                 Fx1	Fy1	Fz1	Tx1	Ty1	Tz1
                 Fx2	Fy2	Fz2	Tx2	Ty2	Tz2
                 ......
                 Fx15	Fy15	Fz15	Tx15	Ty15	Tz15

      where Fx1 ... Fx15 is the evolution of force Fx in the observation
      window, the same for Fy, Fz and the torques; there is a total 
      of 90 features.

8. Missing feature values: None
   
9. Class distribution: percentage of instances per class in each dataset
   -- LP1: 24% normal
           19% collision    
           18% front collision
           39% obstruction
   -- LP2: 43% normal
           13% front collision
           15% back collision
           11% collision to the right
           19% collision to the left
   -- LP3: 43% ok
           19% slightly moved
           32% moved
            6% lost
   -- LP4: 21% normal
           62% collision
           18% obstruction
   -- LP5: 27% normal
           16% bottom collision
           13% bottom obstruction
           29% collision in part
           16% collision in tool
           
10. File format

   -- The file format is as follows:

           <Number of examples> <Example 1> <Example 2> .... <Example N>

      Each example is described as explained in 7.

   -- In order to convert the files to a more standard format,
      the following C program is provided:

      #include <stdio.h>

      char str[128];

      main(int argc,char **argv)
      {
      
         FILE *f1, *f2;
         int i,j,Nex;
         int aux;
      
         f1 = fopen(argv[1],"r");
         f2 = fopen(argv[2],"w");
      
         fscanf(f1,"%d",&Nex);
      
         for(i=0; i<Nex; i++) {
           fscanf(f1,"%s",&str[0]);
           for(j=0; j<90; j++) {
             fscanf(f1,"%d",&aux);
             fprintf(f2,"%d,",aux);
           }
           fprintf(f2,"%s\n",str);                     
         }
      }

