package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.IMatrixNeuralLayer;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class CNNConnection
{
    IMatrixNeuralLayer map;

    List<IMatrixNeuralLayer> lInputMaps = new LinkedList<IMatrixNeuralLayer>();


    public CNNConnection(IMatrixNeuralLayer map, IMatrixNeuralLayer... lInputMaps)
    {
        this.map = map;
        this.lInputMaps = new LinkedList<IMatrixNeuralLayer>(Arrays.asList(lInputMaps));
    }
}
