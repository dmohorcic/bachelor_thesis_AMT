ЪЪ
бЕ
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12unknown8у

conv2d_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_31/kernel
}
$conv2d_31/kernel/Read/ReadVariableOpReadVariableOpconv2d_31/kernel*&
_output_shapes
: *
dtype0
t
conv2d_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_31/bias
m
"conv2d_31/bias/Read/ReadVariableOpReadVariableOpconv2d_31/bias*
_output_shapes
: *
dtype0
|
dense_42/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?* 
shared_namedense_42/kernel
u
#dense_42/kernel/Read/ReadVariableOpReadVariableOpdense_42/kernel* 
_output_shapes
:
?*
dtype0
s
dense_42/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_42/bias
l
!dense_42/bias/Read/ReadVariableOpReadVariableOpdense_42/bias*
_output_shapes	
:*
dtype0
{
dense_43/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X* 
shared_namedense_43/kernel
t
#dense_43/kernel/Read/ReadVariableOpReadVariableOpdense_43/kernel*
_output_shapes
:	X*
dtype0
r
dense_43/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namedense_43/bias
k
!dense_43/bias/Read/ReadVariableOpReadVariableOpdense_43/bias*
_output_shapes
:X*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0

Adam/conv2d_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_31/kernel/m

+Adam/conv2d_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_31/bias/m
{
)Adam/conv2d_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/m*
_output_shapes
: *
dtype0

Adam/dense_42/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*'
shared_nameAdam/dense_42/kernel/m

*Adam/dense_42/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/m* 
_output_shapes
:
?*
dtype0

Adam/dense_42/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_42/bias/m
z
(Adam/dense_42/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_43/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X*'
shared_nameAdam/dense_43/kernel/m

*Adam/dense_43/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/m*
_output_shapes
:	X*
dtype0

Adam/dense_43/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_43/bias/m
y
(Adam/dense_43/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/m*
_output_shapes
:X*
dtype0

Adam/conv2d_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_31/kernel/v

+Adam/conv2d_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_31/bias/v
{
)Adam/conv2d_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_31/bias/v*
_output_shapes
: *
dtype0

Adam/dense_42/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?*'
shared_nameAdam/dense_42/kernel/v

*Adam/dense_42/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/kernel/v* 
_output_shapes
:
?*
dtype0

Adam/dense_42/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_42/bias/v
z
(Adam/dense_42/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_42/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_43/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	X*'
shared_nameAdam/dense_43/kernel/v

*Adam/dense_43/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/kernel/v*
_output_shapes
:	X*
dtype0

Adam/dense_43/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_43/bias/v
y
(Adam/dense_43/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_43/bias/v*
_output_shapes
:X*
dtype0

NoOpNoOp
/
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*з.
valueЭ.BЪ. BУ.

layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
Ќ
+iter

,beta_1

-beta_2
	.decay
/learning_ratemdmemf mg%mh&mivjvkvl vm%vn&vo
 
*
0
1
2
 3
%4
&5
*
0
1
2
 3
%4
&5
­
0non_trainable_variables
1layer_regularization_losses
2layer_metrics

3layers
4metrics
regularization_losses
		variables

trainable_variables
 
\Z
VARIABLE_VALUEconv2d_31/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_31/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
5non_trainable_variables
6layer_metrics
7layer_regularization_losses

8layers
9metrics
regularization_losses
	variables
trainable_variables
 
 
 
­
:non_trainable_variables
;layer_metrics
<layer_regularization_losses

=layers
>metrics
regularization_losses
	variables
trainable_variables
 
 
 
­
?non_trainable_variables
@layer_metrics
Alayer_regularization_losses

Blayers
Cmetrics
regularization_losses
	variables
trainable_variables
 
 
 
­
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses

Glayers
Hmetrics
regularization_losses
	variables
trainable_variables
[Y
VARIABLE_VALUEdense_42/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_42/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
­
Inon_trainable_variables
Jlayer_metrics
Klayer_regularization_losses

Llayers
Mmetrics
!regularization_losses
"	variables
#trainable_variables
[Y
VARIABLE_VALUEdense_43/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_43/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
­
Nnon_trainable_variables
Olayer_metrics
Player_regularization_losses

Qlayers
Rmetrics
'regularization_losses
(	variables
)trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
*
0
1
2
3
4
5

S0
T1
U2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Vtotal
	Wcount
X	variables
Y	keras_api
W
Z
thresholds
[true_positives
\false_positives
]	variables
^	keras_api
W
_
thresholds
`true_positives
afalse_negatives
b	variables
c	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

X	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE

[0
\1

]	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

`0
a1

b	variables
}
VARIABLE_VALUEAdam/conv2d_31/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_31/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_42/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_31/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_31/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_42/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_42/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_43/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_43/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_conv2d_31_inputPlaceholder*0
_output_shapes
:џџџџџџџџџР*
dtype0*%
shape:џџџџџџџџџР
­
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_31_inputconv2d_31/kernelconv2d_31/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 */
f*R(
&__inference_signature_wrapper_48930132
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_31/kernel/Read/ReadVariableOp"conv2d_31/bias/Read/ReadVariableOp#dense_42/kernel/Read/ReadVariableOp!dense_42/bias/Read/ReadVariableOp#dense_43/kernel/Read/ReadVariableOp!dense_43/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp+Adam/conv2d_31/kernel/m/Read/ReadVariableOp)Adam/conv2d_31/bias/m/Read/ReadVariableOp*Adam/dense_42/kernel/m/Read/ReadVariableOp(Adam/dense_42/bias/m/Read/ReadVariableOp*Adam/dense_43/kernel/m/Read/ReadVariableOp(Adam/dense_43/bias/m/Read/ReadVariableOp+Adam/conv2d_31/kernel/v/Read/ReadVariableOp)Adam/conv2d_31/bias/v/Read/ReadVariableOp*Adam/dense_42/kernel/v/Read/ReadVariableOp(Adam/dense_42/bias/v/Read/ReadVariableOp*Adam/dense_43/kernel/v/Read/ReadVariableOp(Adam/dense_43/bias/v/Read/ReadVariableOpConst**
Tin#
!2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__traced_save_48930439
Щ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_31/kernelconv2d_31/biasdense_42/kerneldense_42/biasdense_43/kerneldense_43/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativesAdam/conv2d_31/kernel/mAdam/conv2d_31/bias/mAdam/dense_42/kernel/mAdam/dense_42/bias/mAdam/dense_43/kernel/mAdam/dense_43/bias/mAdam/conv2d_31/kernel/vAdam/conv2d_31/bias/vAdam/dense_42/kernel/vAdam/dense_42/bias/vAdam/dense_43/kernel/vAdam/dense_43/bias/v*)
Tin"
 2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference__traced_restore_48930536ќх
З
з
B__inference_m_35_layer_call_and_return_conditional_losses_48930090

inputs
conv2d_31_48930071
conv2d_31_48930073
dense_42_48930079
dense_42_48930081
dense_43_48930084
dense_43_48930086
identityЂ!conv2d_31/StatefulPartitionedCallЂ dense_42/StatefulPartitionedCallЂ dense_43/StatefulPartitionedCallЋ
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_48930071conv2d_31_48930073*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_489298882#
!conv2d_31/StatefulPartitionedCall
dropout_41/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299212
dropout_41/PartitionedCall
 max_pooling2d_31/PartitionedCallPartitionedCall#dropout_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџT * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_489298672"
 max_pooling2d_31/PartitionedCall
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_flatten_21_layer_call_and_return_conditional_losses_489299412
flatten_21/PartitionedCallЛ
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_42_48930079dense_42_48930081*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_489299602"
 dense_42/StatefulPartitionedCallР
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_48930084dense_43_48930086*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_489299872"
 dense_43/StatefulPartitionedCallч
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0"^conv2d_31/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
ы#
П
B__inference_m_35_layer_call_and_return_conditional_losses_48930197

inputs,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource+
'dense_42_matmul_readvariableop_resource,
(dense_42_biasadd_readvariableop_resource+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource
identityЂ conv2d_31/BiasAdd/ReadVariableOpЂconv2d_31/Conv2D/ReadVariableOpЂdense_42/BiasAdd/ReadVariableOpЂdense_42/MatMul/ReadVariableOpЂdense_43/BiasAdd/ReadVariableOpЂdense_43/MatMul/ReadVariableOpГ
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_31/Conv2D/ReadVariableOpУ
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
paddingVALID*
strides
2
conv2d_31/Conv2DЊ
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOpБ
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
conv2d_31/Relu
dropout_41/IdentityIdentityconv2d_31/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout_41/IdentityЪ
max_pooling2d_31/MaxPoolMaxPooldropout_41/Identity:output:0*/
_output_shapes
:џџџџџџџџџT *
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPoolu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_21/ConstЄ
flatten_21/ReshapeReshape!max_pooling2d_31/MaxPool:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2
flatten_21/ReshapeЊ
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
?*
dtype02 
dense_42/MatMul/ReadVariableOpЄ
dense_42/MatMulMatMulflatten_21/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/MatMulЈ
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_42/BiasAdd/ReadVariableOpІ
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/ReluЉ
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	X*
dtype02 
dense_43/MatMul/ReadVariableOpЃ
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/MatMulЇ
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_43/BiasAdd/ReadVariableOpЅ
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/BiasAdd|
dense_43/SigmoidSigmoiddense_43/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/SigmoidГ
IdentityIdentitydense_43/Sigmoid:y:0!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Ю
g
H__inference_dropout_41_layer_call_and_return_conditional_losses_48929916

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЧ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџЉ :X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs
ЦA

!__inference__traced_save_48930439
file_prefix/
+savev2_conv2d_31_kernel_read_readvariableop-
)savev2_conv2d_31_bias_read_readvariableop.
*savev2_dense_42_kernel_read_readvariableop,
(savev2_dense_42_bias_read_readvariableop.
*savev2_dense_43_kernel_read_readvariableop,
(savev2_dense_43_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop6
2savev2_adam_conv2d_31_kernel_m_read_readvariableop4
0savev2_adam_conv2d_31_bias_m_read_readvariableop5
1savev2_adam_dense_42_kernel_m_read_readvariableop3
/savev2_adam_dense_42_bias_m_read_readvariableop5
1savev2_adam_dense_43_kernel_m_read_readvariableop3
/savev2_adam_dense_43_bias_m_read_readvariableop6
2savev2_adam_conv2d_31_kernel_v_read_readvariableop4
0savev2_adam_conv2d_31_bias_v_read_readvariableop5
1savev2_adam_dense_42_kernel_v_read_readvariableop3
/savev2_adam_dense_42_bias_v_read_readvariableop5
1savev2_adam_dense_43_kernel_v_read_readvariableop3
/savev2_adam_dense_43_bias_v_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*А
valueІBЃB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesФ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesї
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_31_kernel_read_readvariableop)savev2_conv2d_31_bias_read_readvariableop*savev2_dense_42_kernel_read_readvariableop(savev2_dense_42_bias_read_readvariableop*savev2_dense_43_kernel_read_readvariableop(savev2_dense_43_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop2savev2_adam_conv2d_31_kernel_m_read_readvariableop0savev2_adam_conv2d_31_bias_m_read_readvariableop1savev2_adam_dense_42_kernel_m_read_readvariableop/savev2_adam_dense_42_bias_m_read_readvariableop1savev2_adam_dense_43_kernel_m_read_readvariableop/savev2_adam_dense_43_bias_m_read_readvariableop2savev2_adam_conv2d_31_kernel_v_read_readvariableop0savev2_adam_conv2d_31_bias_v_read_readvariableop1savev2_adam_dense_42_kernel_v_read_readvariableop/savev2_adam_dense_42_bias_v_read_readvariableop1savev2_adam_dense_43_kernel_v_read_readvariableop/savev2_adam_dense_43_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ѓ
_input_shapesс
о: : : :
?::	X:X: : : : : : : ::::: : :
?::	X:X: : :
?::	X:X: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?:!

_output_shapes	
::%!

_output_shapes
:	X: 

_output_shapes
:X:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?:!

_output_shapes	
::%!

_output_shapes
:	X: 

_output_shapes
:X:,(
&
_output_shapes
: : 

_output_shapes
: :&"
 
_output_shapes
:
?:!

_output_shapes	
::%!

_output_shapes
:	X: 

_output_shapes
:X:

_output_shapes
: 
С
I
-__inference_dropout_41_layer_call_fn_48930278

inputs
identityв
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299212
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџЉ :X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs
Л'
х
#__inference__wrapped_model_48929861
conv2d_31_input1
-m_35_conv2d_31_conv2d_readvariableop_resource2
.m_35_conv2d_31_biasadd_readvariableop_resource0
,m_35_dense_42_matmul_readvariableop_resource1
-m_35_dense_42_biasadd_readvariableop_resource0
,m_35_dense_43_matmul_readvariableop_resource1
-m_35_dense_43_biasadd_readvariableop_resource
identityЂ%m_35/conv2d_31/BiasAdd/ReadVariableOpЂ$m_35/conv2d_31/Conv2D/ReadVariableOpЂ$m_35/dense_42/BiasAdd/ReadVariableOpЂ#m_35/dense_42/MatMul/ReadVariableOpЂ$m_35/dense_43/BiasAdd/ReadVariableOpЂ#m_35/dense_43/MatMul/ReadVariableOpТ
$m_35/conv2d_31/Conv2D/ReadVariableOpReadVariableOp-m_35_conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$m_35/conv2d_31/Conv2D/ReadVariableOpл
m_35/conv2d_31/Conv2DConv2Dconv2d_31_input,m_35/conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
paddingVALID*
strides
2
m_35/conv2d_31/Conv2DЙ
%m_35/conv2d_31/BiasAdd/ReadVariableOpReadVariableOp.m_35_conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%m_35/conv2d_31/BiasAdd/ReadVariableOpХ
m_35/conv2d_31/BiasAddBiasAddm_35/conv2d_31/Conv2D:output:0-m_35/conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
m_35/conv2d_31/BiasAdd
m_35/conv2d_31/ReluRelum_35/conv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
m_35/conv2d_31/Relu
m_35/dropout_41/IdentityIdentity!m_35/conv2d_31/Relu:activations:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
m_35/dropout_41/Identityй
m_35/max_pooling2d_31/MaxPoolMaxPool!m_35/dropout_41/Identity:output:0*/
_output_shapes
:џџџџџџџџџT *
ksize
*
paddingVALID*
strides
2
m_35/max_pooling2d_31/MaxPool
m_35/flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
m_35/flatten_21/ConstИ
m_35/flatten_21/ReshapeReshape&m_35/max_pooling2d_31/MaxPool:output:0m_35/flatten_21/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2
m_35/flatten_21/ReshapeЙ
#m_35/dense_42/MatMul/ReadVariableOpReadVariableOp,m_35_dense_42_matmul_readvariableop_resource* 
_output_shapes
:
?*
dtype02%
#m_35/dense_42/MatMul/ReadVariableOpИ
m_35/dense_42/MatMulMatMul m_35/flatten_21/Reshape:output:0+m_35/dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
m_35/dense_42/MatMulЗ
$m_35/dense_42/BiasAdd/ReadVariableOpReadVariableOp-m_35_dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02&
$m_35/dense_42/BiasAdd/ReadVariableOpК
m_35/dense_42/BiasAddBiasAddm_35/dense_42/MatMul:product:0,m_35/dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
m_35/dense_42/BiasAdd
m_35/dense_42/ReluRelum_35/dense_42/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
m_35/dense_42/ReluИ
#m_35/dense_43/MatMul/ReadVariableOpReadVariableOp,m_35_dense_43_matmul_readvariableop_resource*
_output_shapes
:	X*
dtype02%
#m_35/dense_43/MatMul/ReadVariableOpЗ
m_35/dense_43/MatMulMatMul m_35/dense_42/Relu:activations:0+m_35/dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
m_35/dense_43/MatMulЖ
$m_35/dense_43/BiasAdd/ReadVariableOpReadVariableOp-m_35_dense_43_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02&
$m_35/dense_43/BiasAdd/ReadVariableOpЙ
m_35/dense_43/BiasAddBiasAddm_35/dense_43/MatMul:product:0,m_35/dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
m_35/dense_43/BiasAdd
m_35/dense_43/SigmoidSigmoidm_35/dense_43/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџX2
m_35/dense_43/Sigmoidж
IdentityIdentitym_35/dense_43/Sigmoid:y:0&^m_35/conv2d_31/BiasAdd/ReadVariableOp%^m_35/conv2d_31/Conv2D/ReadVariableOp%^m_35/dense_42/BiasAdd/ReadVariableOp$^m_35/dense_42/MatMul/ReadVariableOp%^m_35/dense_43/BiasAdd/ReadVariableOp$^m_35/dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2N
%m_35/conv2d_31/BiasAdd/ReadVariableOp%m_35/conv2d_31/BiasAdd/ReadVariableOp2L
$m_35/conv2d_31/Conv2D/ReadVariableOp$m_35/conv2d_31/Conv2D/ReadVariableOp2L
$m_35/dense_42/BiasAdd/ReadVariableOp$m_35/dense_42/BiasAdd/ReadVariableOp2J
#m_35/dense_42/MatMul/ReadVariableOp#m_35/dense_42/MatMul/ReadVariableOp2L
$m_35/dense_43/BiasAdd/ReadVariableOp$m_35/dense_43/BiasAdd/ReadVariableOp2J
#m_35/dense_43/MatMul/ReadVariableOp#m_35/dense_43/MatMul/ReadVariableOp:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input
Д{
њ
$__inference__traced_restore_48930536
file_prefix%
!assignvariableop_conv2d_31_kernel%
!assignvariableop_1_conv2d_31_bias&
"assignvariableop_2_dense_42_kernel$
 assignvariableop_3_dense_42_bias&
"assignvariableop_4_dense_43_kernel$
 assignvariableop_5_dense_43_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count&
"assignvariableop_13_true_positives'
#assignvariableop_14_false_positives(
$assignvariableop_15_true_positives_1'
#assignvariableop_16_false_negatives/
+assignvariableop_17_adam_conv2d_31_kernel_m-
)assignvariableop_18_adam_conv2d_31_bias_m.
*assignvariableop_19_adam_dense_42_kernel_m,
(assignvariableop_20_adam_dense_42_bias_m.
*assignvariableop_21_adam_dense_43_kernel_m,
(assignvariableop_22_adam_dense_43_bias_m/
+assignvariableop_23_adam_conv2d_31_kernel_v-
)assignvariableop_24_adam_conv2d_31_bias_v.
*assignvariableop_25_adam_dense_42_kernel_v,
(assignvariableop_26_adam_dense_42_bias_v.
*assignvariableop_27_adam_dense_43_kernel_v,
(assignvariableop_28_adam_dense_43_bias_v
identity_30ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Є
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*А
valueІBЃB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЪ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesТ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_31_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_31_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ї
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_42_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Ѕ
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_42_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ї
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_43_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_43_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6Ё
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ѓ
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ђ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ў
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ё
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ё
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13Њ
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ћ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ќ
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ћ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Г
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_31_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18Б
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_31_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19В
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_42_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20А
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_42_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21В
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_43_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22А
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_43_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Г
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_31_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Б
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_31_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25В
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_dense_42_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26А
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_dense_42_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27В
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_43_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28А
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_43_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpм
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29Я
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*
_input_shapesx
v: :::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
в
р
B__inference_m_35_layer_call_and_return_conditional_losses_48930026
conv2d_31_input
conv2d_31_48930007
conv2d_31_48930009
dense_42_48930015
dense_42_48930017
dense_43_48930020
dense_43_48930022
identityЂ!conv2d_31/StatefulPartitionedCallЂ dense_42/StatefulPartitionedCallЂ dense_43/StatefulPartitionedCallД
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputconv2d_31_48930007conv2d_31_48930009*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_489298882#
!conv2d_31/StatefulPartitionedCall
dropout_41/PartitionedCallPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299212
dropout_41/PartitionedCall
 max_pooling2d_31/PartitionedCallPartitionedCall#dropout_41/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџT * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_489298672"
 max_pooling2d_31/PartitionedCall
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_flatten_21_layer_call_and_return_conditional_losses_489299412
flatten_21/PartitionedCallЛ
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_42_48930015dense_42_48930017*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_489299602"
 dense_42/StatefulPartitionedCallР
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_48930020dense_43_48930022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_489299872"
 dense_43/StatefulPartitionedCallч
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0"^conv2d_31/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input
ш

+__inference_dense_42_layer_call_fn_48930309

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallњ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_489299602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ?::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
а-
П
B__inference_m_35_layer_call_and_return_conditional_losses_48930168

inputs,
(conv2d_31_conv2d_readvariableop_resource-
)conv2d_31_biasadd_readvariableop_resource+
'dense_42_matmul_readvariableop_resource,
(dense_42_biasadd_readvariableop_resource+
'dense_43_matmul_readvariableop_resource,
(dense_43_biasadd_readvariableop_resource
identityЂ conv2d_31/BiasAdd/ReadVariableOpЂconv2d_31/Conv2D/ReadVariableOpЂdense_42/BiasAdd/ReadVariableOpЂdense_42/MatMul/ReadVariableOpЂdense_43/BiasAdd/ReadVariableOpЂdense_43/MatMul/ReadVariableOpГ
conv2d_31/Conv2D/ReadVariableOpReadVariableOp(conv2d_31_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_31/Conv2D/ReadVariableOpУ
conv2d_31/Conv2DConv2Dinputs'conv2d_31/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
paddingVALID*
strides
2
conv2d_31/Conv2DЊ
 conv2d_31/BiasAdd/ReadVariableOpReadVariableOp)conv2d_31_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_31/BiasAdd/ReadVariableOpБ
conv2d_31/BiasAddBiasAddconv2d_31/Conv2D:output:0(conv2d_31/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
conv2d_31/BiasAdd
conv2d_31/ReluReluconv2d_31/BiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
conv2d_31/Reluy
dropout_41/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout_41/dropout/ConstГ
dropout_41/dropout/MulMulconv2d_31/Relu:activations:0!dropout_41/dropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout_41/dropout/Mul
dropout_41/dropout/ShapeShapeconv2d_31/Relu:activations:0*
T0*
_output_shapes
:2
dropout_41/dropout/Shapeо
/dropout_41/dropout/random_uniform/RandomUniformRandomUniform!dropout_41/dropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
dtype021
/dropout_41/dropout/random_uniform/RandomUniform
!dropout_41/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2#
!dropout_41/dropout/GreaterEqual/yѓ
dropout_41/dropout/GreaterEqualGreaterEqual8dropout_41/dropout/random_uniform/RandomUniform:output:0*dropout_41/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2!
dropout_41/dropout/GreaterEqualЉ
dropout_41/dropout/CastCast#dropout_41/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџЉ 2
dropout_41/dropout/CastЏ
dropout_41/dropout/Mul_1Muldropout_41/dropout/Mul:z:0dropout_41/dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout_41/dropout/Mul_1Ъ
max_pooling2d_31/MaxPoolMaxPooldropout_41/dropout/Mul_1:z:0*/
_output_shapes
:џџџџџџџџџT *
ksize
*
paddingVALID*
strides
2
max_pooling2d_31/MaxPoolu
flatten_21/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
flatten_21/ConstЄ
flatten_21/ReshapeReshape!max_pooling2d_31/MaxPool:output:0flatten_21/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2
flatten_21/ReshapeЊ
dense_42/MatMul/ReadVariableOpReadVariableOp'dense_42_matmul_readvariableop_resource* 
_output_shapes
:
?*
dtype02 
dense_42/MatMul/ReadVariableOpЄ
dense_42/MatMulMatMulflatten_21/Reshape:output:0&dense_42/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/MatMulЈ
dense_42/BiasAdd/ReadVariableOpReadVariableOp(dense_42_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
dense_42/BiasAdd/ReadVariableOpІ
dense_42/BiasAddBiasAdddense_42/MatMul:product:0'dense_42/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/BiasAddt
dense_42/ReluReludense_42/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_42/ReluЉ
dense_43/MatMul/ReadVariableOpReadVariableOp'dense_43_matmul_readvariableop_resource*
_output_shapes
:	X*
dtype02 
dense_43/MatMul/ReadVariableOpЃ
dense_43/MatMulMatMuldense_42/Relu:activations:0&dense_43/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/MatMulЇ
dense_43/BiasAdd/ReadVariableOpReadVariableOp(dense_43_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_43/BiasAdd/ReadVariableOpЅ
dense_43/BiasAddBiasAdddense_43/MatMul:product:0'dense_43/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/BiasAdd|
dense_43/SigmoidSigmoiddense_43/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџX2
dense_43/SigmoidГ
IdentityIdentitydense_43/Sigmoid:y:0!^conv2d_31/BiasAdd/ReadVariableOp ^conv2d_31/Conv2D/ReadVariableOp ^dense_42/BiasAdd/ReadVariableOp^dense_42/MatMul/ReadVariableOp ^dense_43/BiasAdd/ReadVariableOp^dense_43/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2D
 conv2d_31/BiasAdd/ReadVariableOp conv2d_31/BiasAdd/ReadVariableOp2B
conv2d_31/Conv2D/ReadVariableOpconv2d_31/Conv2D/ReadVariableOp2B
dense_42/BiasAdd/ReadVariableOpdense_42/BiasAdd/ReadVariableOp2@
dense_42/MatMul/ReadVariableOpdense_42/MatMul/ReadVariableOp2B
dense_43/BiasAdd/ReadVariableOpdense_43/BiasAdd/ReadVariableOp2@
dense_43/MatMul/ReadVariableOpdense_43/MatMul/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Ю
g
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930263

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ф8?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeН
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ЭЬЬ=2
dropout/GreaterEqual/yЧ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџЉ :X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs
љ	
п
F__inference_dense_42_layer_call_and_return_conditional_losses_48930300

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs

С
'__inference_m_35_layer_call_fn_48930105
conv2d_31_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_m_35_layer_call_and_return_conditional_losses_489300902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input

j
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_48929867

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ы
ќ
B__inference_m_35_layer_call_and_return_conditional_losses_48930051

inputs
conv2d_31_48930032
conv2d_31_48930034
dense_42_48930040
dense_42_48930042
dense_43_48930045
dense_43_48930047
identityЂ!conv2d_31/StatefulPartitionedCallЂ dense_42/StatefulPartitionedCallЂ dense_43/StatefulPartitionedCallЂ"dropout_41/StatefulPartitionedCallЋ
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_31_48930032conv2d_31_48930034*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_489298882#
!conv2d_31/StatefulPartitionedCallЄ
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299162$
"dropout_41/StatefulPartitionedCall
 max_pooling2d_31/PartitionedCallPartitionedCall+dropout_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџT * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_489298672"
 max_pooling2d_31/PartitionedCall
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_flatten_21_layer_call_and_return_conditional_losses_489299412
flatten_21/PartitionedCallЛ
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_42_48930040dense_42_48930042*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_489299602"
 dense_42/StatefulPartitionedCallР
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_48930045dense_43_48930047*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_489299872"
 dense_43/StatefulPartitionedCall
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0"^conv2d_31/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Ж
O
3__inference_max_pooling2d_31_layer_call_fn_48929873

inputs
identityђ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_489298672
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ю
И
'__inference_m_35_layer_call_fn_48930214

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_m_35_layer_call_and_return_conditional_losses_489300512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Џ
I
-__inference_flatten_21_layer_call_fn_48930289

inputs
identityЪ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_flatten_21_layer_call_and_return_conditional_losses_489299412
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџT :W S
/
_output_shapes
:џџџџџџџџџT 
 
_user_specified_nameinputs
ѕ	
п
F__inference_dense_43_layer_call_and_return_conditional_losses_48929987

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџX2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ю
И
'__inference_m_35_layer_call_fn_48930231

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallЉ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_m_35_layer_call_and_return_conditional_losses_489300902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
ѕ	
п
F__inference_dense_43_layer_call_and_return_conditional_losses_48930320

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџX2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџX2	
Sigmoid
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs

С
'__inference_m_35_layer_call_fn_48930066
conv2d_31_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_m_35_layer_call_and_return_conditional_losses_489300512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input
я
f
H__inference_dropout_41_layer_call_and_return_conditional_losses_48929921

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:џџџџџџџџџЉ :X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs
љ	
п
F__inference_dense_42_layer_call_and_return_conditional_losses_48929960

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџ?
 
_user_specified_nameinputs
щ
Р
&__inference_signature_wrapper_48930132
conv2d_31_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference__wrapped_model_489298612
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input
и

р
G__inference_conv2d_31_layer_call_and_return_conditional_losses_48930242

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs


,__inference_conv2d_31_layer_call_fn_48930251

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_489298882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
Р
d
H__inference_flatten_21_layer_call_and_return_conditional_losses_48930284

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџT :W S
/
_output_shapes
:џџџџџџџџџT 
 
_user_specified_nameinputs
ц

+__inference_dense_43_layer_call_fn_48930329

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_489299872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
и

р
G__inference_conv2d_31_layer_call_and_return_conditional_losses_48929888

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂConv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOpЅ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2
Relu 
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџР::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:џџџџџџџџџР
 
_user_specified_nameinputs
я
f
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930268

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:џџџџџџџџџЉ :X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs
Э
f
-__inference_dropout_41_layer_call_fn_48930273

inputs
identityЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:џџџџџџџџџЉ 2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџЉ 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:џџџџџџџџџЉ 
 
_user_specified_nameinputs


B__inference_m_35_layer_call_and_return_conditional_losses_48930004
conv2d_31_input
conv2d_31_48929899
conv2d_31_48929901
dense_42_48929971
dense_42_48929973
dense_43_48929998
dense_43_48930000
identityЂ!conv2d_31/StatefulPartitionedCallЂ dense_42/StatefulPartitionedCallЂ dense_43/StatefulPartitionedCallЂ"dropout_41/StatefulPartitionedCallД
!conv2d_31/StatefulPartitionedCallStatefulPartitionedCallconv2d_31_inputconv2d_31_48929899conv2d_31_48929901*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_conv2d_31_layer_call_and_return_conditional_losses_489298882#
!conv2d_31/StatefulPartitionedCallЄ
"dropout_41/StatefulPartitionedCallStatefulPartitionedCall*conv2d_31/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:џџџџџџџџџЉ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_dropout_41_layer_call_and_return_conditional_losses_489299162$
"dropout_41/StatefulPartitionedCall
 max_pooling2d_31/PartitionedCallPartitionedCall+dropout_41/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџT * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *W
fRRP
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_489298672"
 max_pooling2d_31/PartitionedCall
flatten_21/PartitionedCallPartitionedCall)max_pooling2d_31/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ?* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_flatten_21_layer_call_and_return_conditional_losses_489299412
flatten_21/PartitionedCallЛ
 dense_42/StatefulPartitionedCallStatefulPartitionedCall#flatten_21/PartitionedCall:output:0dense_42_48929971dense_42_48929973*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_42_layer_call_and_return_conditional_losses_489299602"
 dense_42/StatefulPartitionedCallР
 dense_43/StatefulPartitionedCallStatefulPartitionedCall)dense_42/StatefulPartitionedCall:output:0dense_43_48929998dense_43_48930000*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџX*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_dense_43_layer_call_and_return_conditional_losses_489299872"
 dense_43/StatefulPartitionedCall
IdentityIdentity)dense_43/StatefulPartitionedCall:output:0"^conv2d_31/StatefulPartitionedCall!^dense_42/StatefulPartitionedCall!^dense_43/StatefulPartitionedCall#^dropout_41/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџX2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:џџџџџџџџџР::::::2F
!conv2d_31/StatefulPartitionedCall!conv2d_31/StatefulPartitionedCall2D
 dense_42/StatefulPartitionedCall dense_42/StatefulPartitionedCall2D
 dense_43/StatefulPartitionedCall dense_43/StatefulPartitionedCall2H
"dropout_41/StatefulPartitionedCall"dropout_41/StatefulPartitionedCall:a ]
0
_output_shapes
:џџџџџџџџџР
)
_user_specified_nameconv2d_31_input
Р
d
H__inference_flatten_21_layer_call_and_return_conditional_losses_48929941

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ?2

Identity"
identityIdentity:output:0*.
_input_shapes
:џџџџџџџџџT :W S
/
_output_shapes
:џџџџџџџџџT 
 
_user_specified_nameinputs"БL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ф
serving_defaultА
T
conv2d_31_inputA
!serving_default_conv2d_31_input:0џџџџџџџџџР<
dense_430
StatefulPartitionedCall:0џџџџџџџџџXtensorflow/serving/predict:ыб
Л2
layer_with_weights-0
layer-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
*p&call_and_return_all_conditional_losses
q_default_save_signature
r__call__"д/
_tf_keras_sequentialЕ/{"class_name": "Sequential", "name": "m_35", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "m_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_31_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "m_35", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_31_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0006000000284984708, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ѕ


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*s&call_and_return_all_conditional_losses
t__call__"а	
_tf_keras_layerЖ	{"class_name": "Conv2D", "name": "conv2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_31", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}}
ч
regularization_losses
	variables
trainable_variables
	keras_api
*u&call_and_return_all_conditional_losses
v__call__"и
_tf_keras_layerО{"class_name": "Dropout", "name": "dropout_41", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_41", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}

regularization_losses
	variables
trainable_variables
	keras_api
*w&call_and_return_all_conditional_losses
x__call__"ђ
_tf_keras_layerи{"class_name": "MaxPooling2D", "name": "max_pooling2d_31", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_31", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ш
regularization_losses
	variables
trainable_variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"й
_tf_keras_layerП{"class_name": "Flatten", "name": "flatten_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_21", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ї

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*{&call_and_return_all_conditional_losses
|__call__"в
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_42", "trainable": true, "dtype": "float32", "units": 512, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8064}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8064]}}
ї

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
*}&call_and_return_all_conditional_losses
~__call__"в
_tf_keras_layerИ{"class_name": "Dense", "name": "dense_43", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_43", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
П
+iter

,beta_1

-beta_2
	.decay
/learning_ratemdmemf mg%mh&mivjvkvl vm%vn&vo"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
 3
%4
&5"
trackable_list_wrapper
J
0
1
2
 3
%4
&5"
trackable_list_wrapper
Ъ
0non_trainable_variables
1layer_regularization_losses
2layer_metrics

3layers
4metrics
regularization_losses
		variables

trainable_variables
r__call__
q_default_save_signature
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
*:( 2conv2d_31/kernel
: 2conv2d_31/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
5non_trainable_variables
6layer_metrics
7layer_regularization_losses

8layers
9metrics
regularization_losses
	variables
trainable_variables
t__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables
;layer_metrics
<layer_regularization_losses

=layers
>metrics
regularization_losses
	variables
trainable_variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables
@layer_metrics
Alayer_regularization_losses

Blayers
Cmetrics
regularization_losses
	variables
trainable_variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dnon_trainable_variables
Elayer_metrics
Flayer_regularization_losses

Glayers
Hmetrics
regularization_losses
	variables
trainable_variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
#:!
?2dense_42/kernel
:2dense_42/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
­
Inon_trainable_variables
Jlayer_metrics
Klayer_regularization_losses

Llayers
Mmetrics
!regularization_losses
"	variables
#trainable_variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
": 	X2dense_43/kernel
:X2dense_43/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­
Nnon_trainable_variables
Olayer_metrics
Player_regularization_losses

Qlayers
Rmetrics
'regularization_losses
(	variables
)trainable_variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
5
S0
T1
U2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Л
	Vtotal
	Wcount
X	variables
Y	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
Ѓ
Z
thresholds
[true_positives
\false_positives
]	variables
^	keras_api"Щ
_tf_keras_metricЎ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}

_
thresholds
`true_positives
afalse_negatives
b	variables
c	keras_api"Р
_tf_keras_metricЅ{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
.
V0
W1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
[0
\1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
`0
a1"
trackable_list_wrapper
-
b	variables"
_generic_user_object
/:- 2Adam/conv2d_31/kernel/m
!: 2Adam/conv2d_31/bias/m
(:&
?2Adam/dense_42/kernel/m
!:2Adam/dense_42/bias/m
':%	X2Adam/dense_43/kernel/m
 :X2Adam/dense_43/bias/m
/:- 2Adam/conv2d_31/kernel/v
!: 2Adam/conv2d_31/bias/v
(:&
?2Adam/dense_42/kernel/v
!:2Adam/dense_42/bias/v
':%	X2Adam/dense_43/kernel/v
 :X2Adam/dense_43/bias/v
ж2г
B__inference_m_35_layer_call_and_return_conditional_losses_48930004
B__inference_m_35_layer_call_and_return_conditional_losses_48930197
B__inference_m_35_layer_call_and_return_conditional_losses_48930026
B__inference_m_35_layer_call_and_return_conditional_losses_48930168Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ђ2я
#__inference__wrapped_model_48929861Ч
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *7Ђ4
2/
conv2d_31_inputџџџџџџџџџР
ъ2ч
'__inference_m_35_layer_call_fn_48930214
'__inference_m_35_layer_call_fn_48930231
'__inference_m_35_layer_call_fn_48930105
'__inference_m_35_layer_call_fn_48930066Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ё2ю
G__inference_conv2d_31_layer_call_and_return_conditional_losses_48930242Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ж2г
,__inference_conv2d_31_layer_call_fn_48930251Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ю2Ы
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930268
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930263Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
-__inference_dropout_41_layer_call_fn_48930273
-__inference_dropout_41_layer_call_fn_48930278Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
Ж2Г
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_48929867р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
3__inference_max_pooling2d_31_layer_call_fn_48929873р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
ђ2я
H__inference_flatten_21_layer_call_and_return_conditional_losses_48930284Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
з2д
-__inference_flatten_21_layer_call_fn_48930289Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_42_layer_call_and_return_conditional_losses_48930300Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_42_layer_call_fn_48930309Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
№2э
F__inference_dense_43_layer_call_and_return_conditional_losses_48930320Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
е2в
+__inference_dense_43_layer_call_fn_48930329Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
еBв
&__inference_signature_wrapper_48930132conv2d_31_input"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 Ј
#__inference__wrapped_model_48929861 %&AЂ>
7Ђ4
2/
conv2d_31_inputџџџџџџџџџР
Њ "3Њ0
.
dense_43"
dense_43џџџџџџџџџXЙ
G__inference_conv2d_31_layer_call_and_return_conditional_losses_48930242n8Ђ5
.Ђ+
)&
inputsџџџџџџџџџР
Њ ".Ђ+
$!
0џџџџџџџџџЉ 
 
,__inference_conv2d_31_layer_call_fn_48930251a8Ђ5
.Ђ+
)&
inputsџџџџџџџџџР
Њ "!џџџџџџџџџЉ Ј
F__inference_dense_42_layer_call_and_return_conditional_losses_48930300^ 0Ђ-
&Ђ#
!
inputsџџџџџџџџџ?
Њ "&Ђ#

0џџџџџџџџџ
 
+__inference_dense_42_layer_call_fn_48930309Q 0Ђ-
&Ђ#
!
inputsџџџџџџџџџ?
Њ "џџџџџџџџџЇ
F__inference_dense_43_layer_call_and_return_conditional_losses_48930320]%&0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџX
 
+__inference_dense_43_layer_call_fn_48930329P%&0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџXК
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930263n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџЉ 
p
Њ ".Ђ+
$!
0џџџџџџџџџЉ 
 К
H__inference_dropout_41_layer_call_and_return_conditional_losses_48930268n<Ђ9
2Ђ/
)&
inputsџџџџџџџџџЉ 
p 
Њ ".Ђ+
$!
0џџџџџџџџџЉ 
 
-__inference_dropout_41_layer_call_fn_48930273a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџЉ 
p
Њ "!џџџџџџџџџЉ 
-__inference_dropout_41_layer_call_fn_48930278a<Ђ9
2Ђ/
)&
inputsџџџџџџџџџЉ 
p 
Њ "!џџџџџџџџџЉ ­
H__inference_flatten_21_layer_call_and_return_conditional_losses_48930284a7Ђ4
-Ђ*
(%
inputsџџџџџџџџџT 
Њ "&Ђ#

0џџџџџџџџџ?
 
-__inference_flatten_21_layer_call_fn_48930289T7Ђ4
-Ђ*
(%
inputsџџџџџџџџџT 
Њ "џџџџџџџџџ?Р
B__inference_m_35_layer_call_and_return_conditional_losses_48930004z %&IЂF
?Ђ<
2/
conv2d_31_inputџџџџџџџџџР
p

 
Њ "%Ђ"

0џџџџџџџџџX
 Р
B__inference_m_35_layer_call_and_return_conditional_losses_48930026z %&IЂF
?Ђ<
2/
conv2d_31_inputџџџџџџџџџР
p 

 
Њ "%Ђ"

0џџџџџџџџџX
 З
B__inference_m_35_layer_call_and_return_conditional_losses_48930168q %&@Ђ=
6Ђ3
)&
inputsџџџџџџџџџР
p

 
Њ "%Ђ"

0џџџџџџџџџX
 З
B__inference_m_35_layer_call_and_return_conditional_losses_48930197q %&@Ђ=
6Ђ3
)&
inputsџџџџџџџџџР
p 

 
Њ "%Ђ"

0џџџџџџџџџX
 
'__inference_m_35_layer_call_fn_48930066m %&IЂF
?Ђ<
2/
conv2d_31_inputџџџџџџџџџР
p

 
Њ "џџџџџџџџџX
'__inference_m_35_layer_call_fn_48930105m %&IЂF
?Ђ<
2/
conv2d_31_inputџџџџџџџџџР
p 

 
Њ "џџџџџџџџџX
'__inference_m_35_layer_call_fn_48930214d %&@Ђ=
6Ђ3
)&
inputsџџџџџџџџџР
p

 
Њ "џџџџџџџџџX
'__inference_m_35_layer_call_fn_48930231d %&@Ђ=
6Ђ3
)&
inputsџџџџџџџџџР
p 

 
Њ "џџџџџџџџџXё
N__inference_max_pooling2d_31_layer_call_and_return_conditional_losses_48929867RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Щ
3__inference_max_pooling2d_31_layer_call_fn_48929873RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџО
&__inference_signature_wrapper_48930132 %&TЂQ
Ђ 
JЊG
E
conv2d_31_input2/
conv2d_31_inputџџџџџџџџџР"3Њ0
.
dense_43"
dense_43џџџџџџџџџX