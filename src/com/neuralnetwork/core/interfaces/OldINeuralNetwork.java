package com.neuralnetwork.core.interfaces;

import com.neuralnetwork.core.neuron.MNeuron;
import org.ejml.data.DenseMatrix64F;

@Deprecated
public interface OldINeuralNetwork<I,O,N extends INeuron<?>> extends Iterable<N>
{
    public O generateOutput(I input);

    public O generateInducedLocalField(I input);

    public int getNumberOfNeurons();

    public N getNeuron(int neuron);

    public interface IMatrixNeuralNetwork extends OldINeuralNetwork<DenseMatrix64F,DenseMatrix64F,MNeuron>
    {
        public DenseMatrix64F getOutput();

        public int getInputDim();
    }
}
