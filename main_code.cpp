//this file is the code that actually compresses using Discrete Cosine Transform (DCT)

//inclusions for the math processing
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>


//include necessary headers for image processing
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Include headers for plotting
#include "matplotlibcpp.h"

using namespace std;
namespace plt = matplotlibcpp;

//function to preform 2D DCT
vector<vector<double>> dctTransform(const vector<vector<double>>& matrix)
{
  int height = matrix.size(); //get height of the input matrix
  int width = matrix[0].size(); //get width of the input matrix
  vector<vector<double>> dct(height, vector<double>(width, 0.0)); //create 2D vector to store DCT coeff

  for (int u = 0; u < height; ++u)
  {
    for (int v = 0; v < width; ++v)
    {
      double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0; //calc the normalization factor for height
      double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0; //calc the normalization factor for width
      double sum = 0.0; //initialize the sum for DCT coeff

      for (int i = 0; i < height; ++i)
      {
        for (int j = 0; j < width; ++j) //calc DCT coeff for frequency (u,v)
        {
          sum += matrix[i][j] * cos((2 * i + 1) * u * M_PI / (2.0 * height)) * cos((2 * j + 1) * v * M_PI / (2.0 * width));
        }
      }
      dct[u][v] = (2.0 / sqrt(height * width)) * cu * cv * sum; //store calc DCT coeff
    }
  }
  // Log transformation to compress the values
  for (int u = 0; u < height; ++u)
  {
    for (int v = 0; v < width; ++v)
    {
      dct[u][v] = std::copysign(std::log(1.0 + std::abs(dct[u][v])), dct[u][v]);
    }
  }
  return dct; //return the 2D vector of DCT coeffs
}

// Function to perform inverse 2D DCT (IDCT)
vector<vector<double>> idctTransform(const vector<vector<double>>& dct) 
{
  int height = dct.size(); // Get the height of the DCT coefficient matrix
  int width = dct[0].size(); // Get the width of the DCT coefficient matrix
  vector<vector<double>> matrix(height, vector<double>(width, 0.0)); // Create a 2D vector to store the reconstructed image data

  for (int i = 0; i < height; ++i) 
  {
    for (int j = 0; j < width; ++j) 
    {
      double sum = 0.0; // Initialize the sum for the pixel value
      for (int u = 0; u < height; ++u) 
      {
        for (int v = 0; v < width; ++v) 
        {
          double cu = (u == 0) ? 1.0 / sqrt(2.0) : 1.0; // Calculate the normalization factor for u
          double cv = (v == 0) ? 1.0 / sqrt(2.0) : 1.0; // Calculate the normalization factor for v
          // Calculate the pixel value at (i, j)
          sum += cu * cv * dct[u][v] * cos((2 * i + 1) * u * M_PI / (2.0 * height)) * cos((2 * j + 1) * v * M_PI / (2.0 * width));
        }
      }
      matrix[i][j] = (2.0 / sqrt(height * width)) * sum; // Store the calculated pixel value
    }
  }
  return matrix; // Return the 2D vector of reconstructed image data
}
 
// Quantize DCT coefficients
vector<vector<double>> quantizeDCT(const vector<vector<double>>& dct, int quality) {
    vector<vector<int>> quantizationMatrix = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    // Adjust quality factor for scaling
    double qualityFactor = std::max(0.01, (100.0 - quality) / 50.0);

    int height = dct.size();
    int width = dct[0].size();
    vector<vector<double>> quantizedDCT(height, vector<double>(width, 0.0));

    double maxCoeff = 0.0;
    for (const auto& row : dct) {
        for (double val : row) {
            maxCoeff = std::max(maxCoeff, std::abs(val));
        }
    }
    maxCoeff = std::max(maxCoeff, 1e-10);  // Prevent division by zero

    for (int u = 0; u < height; ++u) 
    {
        for (int v = 0; v < width; ++v) 
        {
            double scaledQuantization = quantizationMatrix[u][v] * qualityFactor;
            double normalizedCoeff = std::abs(dct[u][v]) / maxCoeff;
            double threshold = scaledQuantization * (1.0 + std::log1p(normalizedCoeff));

            if (std::abs(dct[u][v]) > threshold) {
                quantizedDCT[u][v] = std::round(dct[u][v] / scaledQuantization) * scaledQuantization;
            }
            else if (std::abs(dct[u][v]) > scaledQuantization * 0.01) {
                quantizedDCT[u][v] = std::copysign(scaledQuantization * 0.01, dct[u][v]);
            }
        }
    }

    // Normalize the quantized DCT coefficients to avoid clipping and improve contrast
    double minVal = 1e10;
    double maxVal = -1e10;

    // Find min and max values
    for (const auto& row : quantizedDCT) {
        for (double val : row) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    // Normalize
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            quantizedDCT[u][v] = (quantizedDCT[u][v] - minVal) / (maxVal - minVal);
        }
    }

    return quantizedDCT;
}

// Dequantize DCT coefficients
vector<vector<double>> dequantizeDCT(const vector<vector<double>>& quantizedDCT, int quality) {
    vector<vector<int>> quantizationMatrix = {
        {16, 11, 10, 16, 24, 40, 51, 61},
        {12, 12, 14, 19, 26, 58, 60, 55},
        {14, 13, 16, 24, 40, 57, 69, 56},
        {14, 17, 22, 29, 51, 87, 80, 62},
        {18, 22, 37, 56, 68, 109, 103, 77},
        {24, 35, 55, 64, 81, 104, 113, 92},
        {49, 64, 78, 87, 103, 121, 120, 101},
        {72, 92, 95, 98, 112, 100, 103, 99}
    };

    // Adjust quality factor for scaling
    double qualityFactor = std::max(0.01, (100.0 - quality) / 50.0);

    int height = quantizedDCT.size();
    int width = quantizedDCT[0].size();
    vector<vector<double>> dequantizedDCT(height, vector<double>(width, 0.0));

    // Dequantization process
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            double scaledQuantization = quantizationMatrix[u][v] * qualityFactor;
            dequantizedDCT[u][v] = quantizedDCT[u][v] * scaledQuantization;
        }
    }

    // Re-normalize the dequantized DCT coefficients to restore the original range
    double minVal = 1e10;
    double maxVal = -1e10;

    // Find min and max values
    for (const auto& row : dequantizedDCT) {
        for (double val : row) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
        }
    }

    // Normalize
    for (int u = 0; u < height; ++u) {
        for (int v = 0; v < width; ++v) {
            dequantizedDCT[u][v] = (dequantizedDCT[u][v] - minVal) / (maxVal - minVal);
        }
    }

    return dequantizedDCT;
}
//plot function
void plotMatrix(const vector<vector<double>>& matrix, const string& title) 
{
  int height = matrix.size();
  int width = matrix[0].size();

  // Find min and max values for proper normalization
  double minVal = matrix[0][0];
  double maxVal = matrix[0][0];
  for (const auto& row : matrix) {
    for (double val : row) {
      minVal = std::min(minVal, val);
      maxVal = std::max(maxVal, val);
    }
  }
  
  // Convert to a 1D vector of unsigned char
  vector<unsigned char> imageData(height * width); 
  for (int i = 0; i < height; ++i) 
  {
    for (int j = 0; j < width; ++j) 
    {
      double normalizedVal = (matrix[i][j] - minVal) / (maxVal - minVal);
      imageData[i * width + j] = static_cast<unsigned char>(std::max(0.0, std::min(255.0, normalizedVal * 255.0))); 
    }
  }

  plt::figure();
  plt::title(title);
  plt::imshow(imageData.data(), height, width, 1, {{"cmap", "magma"}}); // 1 color channel for grayscale
  string filename = title + ".png";
  plt::save(filename);
  cout << "Plot saved as " << filename << endl;

  plt::show();
}

// Function to process an 8x8 block of the image
vector<vector<double>> processBlock(const vector<vector<double>>& block, int quality, vector<vector<double>>& dctCoefficients, vector<vector<double>>& quantizedDCT, vector<vector<double>>& dequantizedDCT) 
{
  // 1. DCT Transform
  vector<vector<double>> dctCoefficients = dctTransform(block);
  

  // 2. Quantization
  vector<vector<double>> quantizedDCT = quantizeDCT(dctCoefficients, quality);
  

  // 3. Dequantization
  vector<vector<double>> dequantizedDCT = dequantizeDCT(quantizedDCT, quality);


  // 4. IDCT Transform
  vector<vector<double>> reconstructedBlock = idctTransform(dequantizedDCT);

  return reconstructedBlock;
}

int main() 
{
  //test for IDCT
  // Create a small test matrix (e.g., 8x8) with some non-zero values
  /*vector<vector<double>> testMatrix = 
  {
    {16.0, 11.0, 10.0, 16.0, 24.0, 40.0, 51.0, 61.0}, 
    {12.0, 12.0, 14.0, 19.0, 26.0, 58.0, 60.0, 55.0}, 
    {14.0, 13.0, 16.0, 24.0, 40.0, 57.0, 69.0, 56.0}, 
    {14.0, 17.0, 22.0, 29.0, 51.0, 87.0, 80.0, 62.0}, 
    {18.0, 22.0, 37.0, 56.0, 68.0, 109.0, 103.0, 77.0}, 
    {24.0, 35.0, 55.0, 64.0, 81.0, 104.0, 113.0, 92.0}, 
    {49.0, 64.0, 78.0, 87.0, 103.0, 121.0, 120.0, 101.0}, 
    {72.0, 92.0, 95.0, 98.0, 112.0, 100.0, 103.0, 99.0}
  };

  //Apply DCT
  vector<vector<double>> resultOne = dctTransform(testMatrix);

  // Print the result
  cout << "DCT coeff:" << endl;
  for (int m = 0; m < 8; ++m) 
  {
    for (int n = 0; n < 8; ++n) 
    {
      std::cout << resultOne[m][n] << " ";
    }
    std::cout << std::endl;
  }
  
  // Apply IDCT
  vector<vector<double>> result = idctTransform(resultOne);

  // Print the result
  cout << "IDCT coeff:" << endl;
  for (int i = 0; i < 8; ++i) 
  {
    for (int j = 0; j < 8; ++j) 
    {
      std::cout << result[i][j] << " ";
    }
    std::cout << std::endl;
  }*/
  //image section
  
  //step 1: call in image that the user uploaded
  string filename = "kirby_8x8.png"; //test image 
  
  //step 2: load the image data
  int width, height, channels;
  unsigned char* imageData = stbi_load(filename.c_str(), &width, &height, &channels, 0);
  if (imageData == nullptr) 
  {
    cerr << "Error loading image: " << stbi_failure_reason() << endl;
    return 1;
  }
  
  //checks for image dimensions and data size
  size_t imageDataSize = width * height * channels * sizeof(unsigned char);
  if (width <= 0 || height <= 0 || channels <= 0 || (width * height * channels) > imageDataSize) 
  {
    cerr << "Error: Invalid image data." << endl;
    return 1;
  }
  
  //step 3: convert the image data to a 2D vector of doubles (grayscale)
  vector<vector<vector<double>>> imageMatrices(channels);
  for (int c = 0; c < channels; ++c) 
  {
    imageMatrices[c] = vector<vector<double>>(height, vector<double>(width));
    for (int i = 0; i < height; ++i)
    {
      for (int j = 0; j < width; ++j) 
      {
        // Calculate the index into the imageData array
        int index = i * width * channels + j * channels + c; 
        imageMatrices[c][i][j] = static_cast<double>(imageData[index]) / 255.0; 
      }
    }
  }
  
  stbi_image_free(imageData); //free the original image data
  
  //step 5: quantize DCT coefficients and adjust quality as needed
  int quality = 50; // You can get this from the user or another source
  //cout << "Enter desired quality (0-100): ";
  //cin >> quality;

  //initialize reconstructedChannels 
  vector<vector<vector<double>>> reconstructedChannels(channels);
  // Store matrices to plot once per channel
  vector<vector<double>> dctCoefficients(8, vector<double>(8));
  vector<vector<double>> quantizedDCT(8, vector<double>(8));
  vector<vector<double>> dequantizedDCT(8, vector<double>(8));
  vector<vector<double>> reconstructedBlock(8, vector<double>(8));
  
  for (int c = 0; c < channels; ++c)
  {
    reconstructedChannels[c] = vector<vector<double>>(height, vector<double>(width,0.0));
    vector<vector<double>> dctCoefficients(8, vector<double>(8));
    vector<vector<double>> quantizedDCT(8, vector<double>(8));
    vector<vector<double>> dequantizedDCT(8, vector<double>(8));
    vector<vector<double>> reconstructedBlock(8, vector<double>(8));
    
    // Process blocks per channel
    for (int i = 0; i < height; i += 8) 
    {
      for (int j = 0; j < width; j += 8) 
      {
        // Extract an 8x8 block from the image
        vector<vector<double>> block(8, vector<double>(8));
        for (int u = 0; u < 8; ++u) 
        {
          for (int v = 0; v < 8; ++v) 
          {
            // Handle boundary conditions: If the block goes outside the image,
            // use the edge pixel value.
            int imgU = std::min(i + u, height - 1);
            int imgV = std::min(j + v, width - 1);
            block[u][v] = imageMatrices[c][imgU][imgV]; 
          }
        }

        // Process the block for this channel
        vector<vector<double>> reconstructedBlock = processBlock(block, quality, dctCoefficients, quantizedDCT, dequantizedDCT);
        // Copy the processed block back into reconstructed channel
        for (int u = 0; u < 8; ++u) 
        {
          for (int v = 0; v < 8; ++v) 
          {
            int imgU = std::min(i + u, height - 1);
            int imgV = std::min(j + v, width - 1);
            reconstructedChannels[c][imgU][imgV] = processedBlock[u][v];          
          }
        }
        //if this is the last block, plot
        if (c == channels - 1 && i == height - 8 && j == width - 8)
        {
          // Plot the matrices once per channel
        plotMatrix(dctCoefficients, "DCT Coefficients");
        plotMatrix(quantizedDCT, "Quantized DCT");
        plotMatrix(dequantizedDCT, "De-Quantized DCT");
        plotMatrix(reconstructedBlock, "Reconstructed Block");
        }
      }
    }
  }
  //step 8: convert back to an unsigned char format for saving
  vector<unsigned char> outputData(height * width * channels);
  for (int c = 0; c < channels; ++c)
  {
    for (int i = 0; i < height; ++i) 
    {
      for (int j = 0; j < width; ++j) 
      {
        outputData[i * width * channels + j * channels + c] = static_cast<unsigned char>(reconstructedChannels[c][i][j] * 255.0);
      }
    }
  }
  //step 9: save the reconstructed image
  string outputFilename = "reconstructed_" + filename;
  if (stbi_write_png(outputFilename.c_str(), width, height, channels, outputData.data(), width * channels) == 0) 
  {
    cerr << "Error saving image." << endl;
    return 1;
  }
  cout << "Reconstructed image saved as " << outputFilename << endl;
  cout << "Image dimensions: " << width << " x " << height << endl;
  cout << "Number of channels: " << channels << endl;
  
  return 0;
}
