package com.neuralnetwork.cnn;

import com.neuralnetwork.core.interfaces.ICnnMap;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

public class CNNConnection
{
    ICnnMap map;

    List<ICnnMap> lInputMaps = new LinkedList<ICnnMap>();


    public CNNConnection(ICnnMap map, ICnnMap... lInputMaps)
    {
        this.map = map;
        this.lInputMaps = new LinkedList<ICnnMap>(Arrays.asList(lInputMaps));
    }
}
