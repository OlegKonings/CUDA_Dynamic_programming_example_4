CUDA_Dynamic_programming_example_4
==================================

Another CUDA implementations of a DP problem

A CPU and GPU version of the Top Coder problem 'LinearKingdomParkingLot';

http://community.topcoder.com/stat?c=problem_statement&pm=10982&rd=14283

This version first was tested on the smaller sample test sets, but also has a function which generates random data exit sets for comparison.

Running time is 2*((num_cars+1)*(num_cars+1)*num_cars+1)+(num_cars+1)*(num_cars+1).

____
<table>
<tr>
    <th>Num Cars</th><th>Iterations</th><th>CPU time</th><th>GPU time</th><th>CUDA Speedup</th>
</tr>

  <tr>
    <td>300</td><td>54,632,403</td><td> 1078 ms</td><td>  28 ms</td><td> 38.5x</td>
  </tr>
  <tr>
    <td>400</td><td>129,123,203 </td><td> 2529 ms</td><td>  65 ms</td><td> 39.21x</td>
  </tr>
</table>  
___

NOTE: All CUDA GPU times include all device memsets, host-device memory copies and device-host memory copies.

CPU= Intel I-7 3770K 3.5 Ghz with 3.9 Ghz target

GPU= Tesla K20c 5GB

Windows 7 Ultimate x64

Visual Studio 2010 x64



Overall the CUDA speedup is 11x-40x faster than a very fast optimized CPU implementation. 

This type of algorithm will NOT run faster in MATLAB. This class of Dynamic Programming problem implemented in CUDA will run faster than any other implementation on any single PC which costs less than $4000. If you can beat it for 400 cars, please let me know. 


