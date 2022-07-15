// CS 61C Fall 2014 Project 3

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"
#include <stdio.h>

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	if (featureWidth % 3)
	{
		for (int x = 0; x < imageWidth; x++)
		{
			for (int y = 0; y < imageHeight; y++)
			{
				if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
				{
					depth[y * imageWidth + x] = 0;
					continue;
				}

				float minimumSquaredDifference = -1;
				int minimumDy = 0;
				int minimumDx = 0;

				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
					{
						if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
						{
							continue;
						}

						float squaredDifference = 0;
						__m128 v_squaredDifference = _mm_setzero_ps();

					/*
						for (int boxX = -featureWidth; boxX < -featureWidth + ((((featureWidth * 2) + 1) / 16) * 16); boxX += 16)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 4]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 4]));
								__m128 a2 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 8]));
								__m128 b2 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 8]));
								__m128 a3 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 12]));
								__m128 b3 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 12]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								a2 = _mm_sub_ps(a2, b2);
								a2 = _mm_mul_ps(a2, a2);
								a3 = _mm_sub_ps(a3, b3);
								a3 = _mm_mul_ps(a3, a3);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a2);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a3);
							}
						}

						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 16) * 16); boxX < -featureWidth + ((((featureWidth * 2) + 1) / 12) * 12); boxX += 12)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 4]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 4]));
								__m128 a2 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 8]));
								__m128 b2 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 8]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								a2 = _mm_sub_ps(a2, b2);
								a2 = _mm_mul_ps(a2, a2);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a2);
							}
						}
*/
						for (int boxX = -featureWidth; boxX < -featureWidth + ((((featureWidth * 2) + 1) / 8) * 8); boxX += 8)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 4]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 4]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
							}
						}

						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 8) * 8); boxX < -featureWidth + ((((featureWidth * 2) + 1) / 4) * 4); boxX += 4)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
							}
						}

						v_squaredDifference = _mm_hadd_ps(v_squaredDifference, v_squaredDifference);
						v_squaredDifference = _mm_hadd_ps(v_squaredDifference, v_squaredDifference);
						_mm_store_ss(&squaredDifference, v_squaredDifference);

						if (squaredDifference > minimumSquaredDifference && minimumSquaredDifference != -1)
							continue;

						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 4) * 4); boxX <= featureWidth; boxX++)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
								squaredDifference += difference * difference;
							}
						}

						if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
						{
							minimumSquaredDifference = squaredDifference;
							minimumDx = dx;
							minimumDy = dy;
						}
					}
				}

				if (minimumSquaredDifference != -1)
				{
					if (maximumDisplacement == 0)
					{
						depth[y * imageWidth + x] = 0;
					}
					else
					{
						depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
					}
				}
				else
				{
					depth[y * imageWidth + x] = 0;
				}
			}
		}
	}
	else
	{
		for (int x = 0; x < imageWidth; x++)
		{
			for (int y = 0; y < imageHeight; y++)
			{
				if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
				{
					depth[y * imageWidth + x] = 0;
					continue;
				}

				float minimumSquaredDifference = -1;
				int minimumDy = 0;
				int minimumDx = 0;

				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
					{
						if (y + dy - featureHeight < 0 || y + dy + featureHeight >= imageHeight || x + dx - featureWidth < 0 || x + dx + featureWidth >= imageWidth)
						{
							continue;
						}

						float squaredDifference = 0;
						__m128 v_squaredDifference = _mm_setzero_ps();

						float baby[4] = {1, 1, 1, 0};
/*
						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 16) * 16); boxX < -featureWidth + ((((featureWidth * 2) + 1) / 12) * 12); boxX += 12)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 4]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 4]));
								__m128 a2 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 8]));
								__m128 b2 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 8]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								a2 = _mm_sub_ps(a2, b2);
								a2 = _mm_mul_ps(a2, a2);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a2);
							}
						}

						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 12) * 12); boxX < -featureWidth + ((((featureWidth * 2) + 1) / 9) * 9); boxX += 9)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 3]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 3]));
								__m128 a2 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 6]));
								__m128 b2 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 6]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								a2 = _mm_sub_ps(a2, b2);
								a2 = _mm_mul_ps(a2, a2);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a2);
							}
						}*/
						for (int boxX = -featureWidth; boxX < -featureWidth + ((((featureWidth * 2) + 1) / 6) * 6); boxX += 6)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								__m128 a1 = _mm_loadu_ps(&(left[leftY * imageWidth + leftX + 3]));
								__m128 b1 = _mm_loadu_ps(&(right[rightY * imageWidth + rightX + 3]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								a1 = _mm_sub_ps(a1, b1);
								a1 = _mm_mul_ps(a1, a1);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a1);
							}
						}
						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 6) * 6); boxX < -featureWidth + ((((featureWidth * 2) + 1) / 3) * 3); boxX += 3)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								__m128 a = _mm_loadu_ps(&(left[leftY * imageWidth + leftX]));
								__m128 b = _mm_loadu_ps(&(right[rightY * imageWidth + rightX]));
								a = _mm_sub_ps(a, b);
								a = _mm_mul_ps(a, a);
								v_squaredDifference = _mm_add_ps(v_squaredDifference, a);
							}
						}

						__m128 c = _mm_loadu_ps(baby);
						v_squaredDifference = _mm_mul_ps(v_squaredDifference, c);
						v_squaredDifference = _mm_hadd_ps(v_squaredDifference, v_squaredDifference);
						v_squaredDifference = _mm_hadd_ps(v_squaredDifference, v_squaredDifference);
						_mm_store_ss(&squaredDifference, v_squaredDifference);

						if (squaredDifference > minimumSquaredDifference && minimumSquaredDifference != -1)
							continue;

						for (int boxX = -featureWidth + ((((featureWidth * 2) + 1) / 3) * 3); boxX <= featureWidth; boxX++)
						{
							for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
							{
								int leftX = x + boxX;
								int leftY = y + boxY;
								int rightX = x + dx + boxX;
								int rightY = y + dy + boxY;

								float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
								squaredDifference += difference * difference;
							}
						}

						if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < displacementNaive(minimumDx, minimumDy))) || (minimumSquaredDifference > squaredDifference))
						{
							minimumSquaredDifference = squaredDifference;
							minimumDx = dx;
							minimumDy = dy;
						}
					}
				}

				if (minimumSquaredDifference != -1)
				{
					if (maximumDisplacement == 0)
					{
						depth[y * imageWidth + x] = 0;
					}
					else
					{
						depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
					}
				}
				else
				{
					depth[y * imageWidth + x] = 0;
				}
			}
		}
	}	
}
