clear;
clc;
close all;
%==========================================================================
% question 1
%==========================================================================
% given P(light) = 0.25, P(camera) = 0.2, and P(light,camera) = 0.05
% P(camera|light) = P(camera,light) / P(light) = 0.05 / 0.25 = 0.2


%==========================================================================
% question 2
%==========================================================================
% P(A,B,C,D,E,G)
% = P(G|A,B,C,D,E) * P(A,B,C,D,E)
% = P(G|E) * P(E|A,B,C,D) * P(A,B,C,D)
% = 0.5 * P(E|C) * P(D|A,B,C) * P(A.B.C)
% = 0.5 * 0.1 * P(D|B,C) * P(A.B.C)
% = 0.5 * 0.1 * 0.5 * P(C|A.B) * P(A.B)
% = 0.5 * 0.1 * 0.5 * P(C|A) * P(B|A) * P(A)
% = 0.5 * 0.1 * 0.5 * 0.8 * 0.1 * 0.5
% = 0.001


% P(B,C)
% = P(B,C,A) + P(B,C,A')
% = P(B|C,A) * P(C,A) + P(B|C,A') * P(C,A')
% = P(B|A) * P(C,A) + P(B|A') * P(C,A')
% = 0.1 * P(C|A) * P(A) + 0.5 * P(C|A') * P(A')
% = 0.1 * 0.8 * 0.5 + 0.5 * 0.1 * 0.5
% = 0.001


%==========================================================================
% question 3
%==========================================================================
% P(A,B',C,D,E',F,I) follows a similar pattern to previous question
% = P(I|F) * P(F|D) * P(E'|C) * P(D|B',C) * P(C|A) * P(B?|A) * P(A)


%==========================================================================
% question 4
%==========================================================================
% Pitch at 14K, w1 = kitten, W2 = puppy
% P(W1|14K) = P(14K|W1) * P(W1) / P(14K)
% P(W2|14K) = P(14K|W2) * P(W2) / P(14K)
% Compare P(W1|14K) with P(W2|14K), the denominator can be cancelled out
% It reduces to P(14K|W1) * P(W1) ? P(14K|W2) * P(W2)
% Looking at the graph
% P(14K|W1) = 0.1, P(W1) = 3 / 30 = 0.1
% P(14K|W2) = 0.06, P(W2) = 27 / 30 = 0.9
% P(14K|W1) * P(W1) = 0.1 * 0.1 = 0.01
% P(14K|W2) * P(W2) = 0.06 * 0.9 = 0.054
% It's more likely to be a puppy because the there is way more puppies that
% kittens


