package com.neuralnetwork.core.interfaces;

import org.ejml.data.DenseMatrix64F;

/**
 * Works in conjunction with {@link com.neuralnetwork.core.backprop.NNBackprop}
 *
 * Assumes a specific relationship between input, output, neuron weights
 *
 * Example:
 * Input  Neuron Weights       Output
 * I0                          1
 * I1     N1: w10 w11 w12 w13  O1
 * I2     N2: w20 w21 w22 w23  O2
 * I3     N3: w30 w31 w32 w33  O3
 *
 * 1. rows in weight matrix are a neuron's weights
 * 2. weights are in specific order:
 *    bias comes first (column 1),
 *    the second weight (column 2) maps to the first neuron,
 *    the third weight (column 3) maps the second neuron,
 *    etc.
 * 3. the first value in the input is always 1
 *    (maps to the bias)
 * 4. output (including induced local field) is in specific order:
 *    first value is always 1 (maps to bias),
 *    second value is output of first neuron,
 *    third value is output of second neuron,
 *    etc.
 * 5. input/outputs are 1D arrays but don't have to be.
 *
 * For simplicity we also assume that each neuron has the same activation function
 * (although it doesn't have to be).
 *
 * Again, we also arbitrarily define the first weight in a neuron to be the bias.
 */
public interface INeuralLayer
{
    /**
     * We use the convention that the bias is at the beginning of the array i.e.,
     * input[0] = +1 and
     * output[0] = +1
     *
     * @param input
     * @return
     */
    public double[] generateOutput(double[] input);

    /**
     * We use the convention that the bias is at the beginning of the array i.e.,
     * input[0] = +1 and
     * output[0] = +1
     *
     * @param input
     * @return
     */
    public double[] generateInducedLocalField(double[] input);

    public int getInputDim();

    public int getOutputDim();

    public int getNumberOfNeurons();

    /**
     * Unfortunately, for this to work, the matrix needs to be in a specific format:
     * Neuron i's weights are in row i.
     *
     * Example:
     * N1: w10 w11 w12 w13
     * N2: w20 w21 w22 w23
     * N3: w30 w31 w32 w33
     *
     * @return
     */
    public DenseMatrix64F getWeightMatrix();

    /**
     * For simplicity, each neuron has the same impulse function.
     *
     * @return
     */
    public IActivationFunction.IDifferentiableFunction getImpulseFunction();
}
