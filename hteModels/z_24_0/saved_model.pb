??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
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
?
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
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
: *
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
: *
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
_output_shapes
:@*
dtype0
{
dense_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X* 
shared_namedense_15/kernel
t
#dense_15/kernel/Read/ReadVariableOpReadVariableOpdense_15/kernel*
_output_shapes
:	?X*
dtype0
r
dense_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namedense_15/bias
k
!dense_15/bias/Read/ReadVariableOpReadVariableOpdense_15/bias*
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
?
Adam/conv2d_25/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_25/kernel/m
?
+Adam/conv2d_25/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_25/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_25/bias/m
{
)Adam/conv2d_25/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_26/kernel/m
?
+Adam/conv2d_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/m
{
)Adam/conv2d_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X*'
shared_nameAdam/dense_15/kernel/m
?
*Adam/dense_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/m*
_output_shapes
:	?X*
dtype0
?
Adam/dense_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_15/bias/m
y
(Adam/dense_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/m*
_output_shapes
:X*
dtype0
?
Adam/conv2d_25/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_25/kernel/v
?
+Adam/conv2d_25/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_25/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_25/bias/v
{
)Adam/conv2d_25/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_25/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_26/kernel/v
?
+Adam/conv2d_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2d_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_26/bias/v
{
)Adam/conv2d_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_26/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?X*'
shared_nameAdam/dense_15/kernel/v
?
*Adam/dense_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/kernel/v*
_output_shapes
:	?X*
dtype0
?
Adam/dense_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*%
shared_nameAdam/dense_15/bias/v
y
(Adam/dense_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_15/bias/v*
_output_shapes
:X*
dtype0

NoOpNoOp
?3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?3
value?3B?3 B?3
?
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
R
#regularization_losses
$	variables
%trainable_variables
&	keras_api
R
'regularization_losses
(	variables
)trainable_variables
*	keras_api
R
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemxmymzm{/m|0m}v~vv?v?/v?0v?
 
*
0
1
2
3
/4
05
*
0
1
2
3
/4
05
?
:layer_metrics

regularization_losses

;layers
	variables
<non_trainable_variables
=layer_regularization_losses
>metrics
trainable_variables
 
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
?layer_metrics
regularization_losses

@layers
	variables
Anon_trainable_variables
Blayer_regularization_losses
Cmetrics
trainable_variables
 
 
 
?
Dlayer_metrics
regularization_losses

Elayers
	variables
Fnon_trainable_variables
Glayer_regularization_losses
Hmetrics
trainable_variables
 
 
 
?
Ilayer_metrics
regularization_losses

Jlayers
	variables
Knon_trainable_variables
Llayer_regularization_losses
Mmetrics
trainable_variables
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_26/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
Nlayer_metrics
regularization_losses

Olayers
 	variables
Pnon_trainable_variables
Qlayer_regularization_losses
Rmetrics
!trainable_variables
 
 
 
?
Slayer_metrics
#regularization_losses

Tlayers
$	variables
Unon_trainable_variables
Vlayer_regularization_losses
Wmetrics
%trainable_variables
 
 
 
?
Xlayer_metrics
'regularization_losses

Ylayers
(	variables
Znon_trainable_variables
[layer_regularization_losses
\metrics
)trainable_variables
 
 
 
?
]layer_metrics
+regularization_losses

^layers
,	variables
_non_trainable_variables
`layer_regularization_losses
ametrics
-trainable_variables
[Y
VARIABLE_VALUEdense_15/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_15/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
blayer_metrics
1regularization_losses

clayers
2	variables
dnon_trainable_variables
elayer_regularization_losses
fmetrics
3trainable_variables
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
8
0
1
2
3
4
5
6
7
 
 

g0
h1
i2
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
	jtotal
	kcount
l	variables
m	keras_api
W
n
thresholds
otrue_positives
pfalse_positives
q	variables
r	keras_api
W
s
thresholds
ttrue_positives
ufalse_negatives
v	variables
w	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

l	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

q	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

t0
u1

v	variables
}
VARIABLE_VALUEAdam/conv2d_25/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_25/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_25/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_26/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_26/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_15/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_15/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_conv2d_25_inputPlaceholder*0
_output_shapes
:??????????*
dtype0*%
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_25_inputconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_15/kerneldense_15/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_38891129
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOp#dense_15/kernel/Read/ReadVariableOp!dense_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp+Adam/conv2d_25/kernel/m/Read/ReadVariableOp)Adam/conv2d_25/bias/m/Read/ReadVariableOp+Adam/conv2d_26/kernel/m/Read/ReadVariableOp)Adam/conv2d_26/bias/m/Read/ReadVariableOp*Adam/dense_15/kernel/m/Read/ReadVariableOp(Adam/dense_15/bias/m/Read/ReadVariableOp+Adam/conv2d_25/kernel/v/Read/ReadVariableOp)Adam/conv2d_25/bias/v/Read/ReadVariableOp+Adam/conv2d_26/kernel/v/Read/ReadVariableOp)Adam/conv2d_26/bias/v/Read/ReadVariableOp*Adam/dense_15/kernel/v/Read/ReadVariableOp(Adam/dense_15/bias/v/Read/ReadVariableOpConst**
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
GPU2*0J 8? **
f%R#
!__inference__traced_save_38891474
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/biasdense_15/kerneldense_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativesAdam/conv2d_25/kernel/mAdam/conv2d_25/bias/mAdam/conv2d_26/kernel/mAdam/conv2d_26/bias/mAdam/dense_15/kernel/mAdam/dense_15/bias/mAdam/conv2d_25/kernel/vAdam/conv2d_25/bias/vAdam/conv2d_26/kernel/vAdam/conv2d_26/bias/vAdam/dense_15/kernel/vAdam/dense_15/bias/v*)
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
GPU2*0J 8? *-
f(R&
$__inference__traced_restore_38891571??
?
I
-__inference_flatten_15_layer_call_fn_38891344

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_15_layer_call_and_return_conditional_losses_388909592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$@:W S
/
_output_shapes
:?????????$@
 
_user_specified_nameinputs
?	
?
F__inference_dense_15_layer_call_and_return_conditional_losses_38890978

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????X2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?A
?
!__inference__traced_save_38891474
file_prefix/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop.
*savev2_dense_15_kernel_read_readvariableop,
(savev2_dense_15_bias_read_readvariableop(
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
2savev2_adam_conv2d_25_kernel_m_read_readvariableop4
0savev2_adam_conv2d_25_bias_m_read_readvariableop6
2savev2_adam_conv2d_26_kernel_m_read_readvariableop4
0savev2_adam_conv2d_26_bias_m_read_readvariableop5
1savev2_adam_dense_15_kernel_m_read_readvariableop3
/savev2_adam_dense_15_bias_m_read_readvariableop6
2savev2_adam_conv2d_25_kernel_v_read_readvariableop4
0savev2_adam_conv2d_25_bias_v_read_readvariableop6
2savev2_adam_conv2d_26_kernel_v_read_readvariableop4
0savev2_adam_conv2d_26_bias_v_read_readvariableop5
1savev2_adam_dense_15_kernel_v_read_readvariableop3
/savev2_adam_dense_15_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableop*savev2_dense_15_kernel_read_readvariableop(savev2_dense_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop2savev2_adam_conv2d_25_kernel_m_read_readvariableop0savev2_adam_conv2d_25_bias_m_read_readvariableop2savev2_adam_conv2d_26_kernel_m_read_readvariableop0savev2_adam_conv2d_26_bias_m_read_readvariableop1savev2_adam_dense_15_kernel_m_read_readvariableop/savev2_adam_dense_15_bias_m_read_readvariableop2savev2_adam_conv2d_25_kernel_v_read_readvariableop0savev2_adam_conv2d_25_bias_v_read_readvariableop2savev2_adam_conv2d_26_kernel_v_read_readvariableop0savev2_adam_conv2d_26_bias_v_read_readvariableop1savev2_adam_dense_15_kernel_v_read_readvariableop/savev2_adam_dense_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *,
dtypes"
 2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:	?X:X: : : : : : : ::::: : : @:@:	?X:X: : : @:@:	?X:X: 2(
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
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?X: 
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
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?X: 

_output_shapes
:X:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?X: 

_output_shapes
:X:

_output_shapes
: 
?

?
G__inference_conv2d_26_layer_call_and_return_conditional_losses_38890906

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????I@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T 
 
_user_specified_nameinputs
?	
?
F__inference_dense_15_layer_call_and_return_conditional_losses_38891355

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????X2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_26_layer_call_fn_38891306

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_388909062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????T 
 
_user_specified_nameinputs
?+
?
#__inference__wrapped_model_38890809
conv2d_25_input1
-m_24_conv2d_25_conv2d_readvariableop_resource2
.m_24_conv2d_25_biasadd_readvariableop_resource1
-m_24_conv2d_26_conv2d_readvariableop_resource2
.m_24_conv2d_26_biasadd_readvariableop_resource0
,m_24_dense_15_matmul_readvariableop_resource1
-m_24_dense_15_biasadd_readvariableop_resource
identity??%m_24/conv2d_25/BiasAdd/ReadVariableOp?$m_24/conv2d_25/Conv2D/ReadVariableOp?%m_24/conv2d_26/BiasAdd/ReadVariableOp?$m_24/conv2d_26/Conv2D/ReadVariableOp?$m_24/dense_15/BiasAdd/ReadVariableOp?#m_24/dense_15/MatMul/ReadVariableOp?
$m_24/conv2d_25/Conv2D/ReadVariableOpReadVariableOp-m_24_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$m_24/conv2d_25/Conv2D/ReadVariableOp?
m_24/conv2d_25/Conv2DConv2Dconv2d_25_input,m_24/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingVALID*
strides
2
m_24/conv2d_25/Conv2D?
%m_24/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp.m_24_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%m_24/conv2d_25/BiasAdd/ReadVariableOp?
m_24/conv2d_25/BiasAddBiasAddm_24/conv2d_25/Conv2D:output:0-m_24/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
m_24/conv2d_25/BiasAdd?
m_24/conv2d_25/ReluRelum_24/conv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
m_24/conv2d_25/Relu?
m_24/dropout_25/IdentityIdentity!m_24/conv2d_25/Relu:activations:0*
T0*0
_output_shapes
:?????????? 2
m_24/dropout_25/Identity?
m_24/max_pooling2d_25/MaxPoolMaxPool!m_24/dropout_25/Identity:output:0*/
_output_shapes
:?????????T *
ksize
*
paddingVALID*
strides
2
m_24/max_pooling2d_25/MaxPool?
$m_24/conv2d_26/Conv2D/ReadVariableOpReadVariableOp-m_24_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02&
$m_24/conv2d_26/Conv2D/ReadVariableOp?
m_24/conv2d_26/Conv2DConv2D&m_24/max_pooling2d_25/MaxPool:output:0,m_24/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@*
paddingVALID*
strides
2
m_24/conv2d_26/Conv2D?
%m_24/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp.m_24_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02'
%m_24/conv2d_26/BiasAdd/ReadVariableOp?
m_24/conv2d_26/BiasAddBiasAddm_24/conv2d_26/Conv2D:output:0-m_24/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@2
m_24/conv2d_26/BiasAdd?
m_24/conv2d_26/ReluRelum_24/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I@2
m_24/conv2d_26/Relu?
m_24/dropout_26/IdentityIdentity!m_24/conv2d_26/Relu:activations:0*
T0*/
_output_shapes
:?????????I@2
m_24/dropout_26/Identity?
m_24/max_pooling2d_26/MaxPoolMaxPool!m_24/dropout_26/Identity:output:0*/
_output_shapes
:?????????$@*
ksize
*
paddingVALID*
strides
2
m_24/max_pooling2d_26/MaxPool
m_24/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
m_24/flatten_15/Const?
m_24/flatten_15/ReshapeReshape&m_24/max_pooling2d_26/MaxPool:output:0m_24/flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
m_24/flatten_15/Reshape?
#m_24/dense_15/MatMul/ReadVariableOpReadVariableOp,m_24_dense_15_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02%
#m_24/dense_15/MatMul/ReadVariableOp?
m_24/dense_15/MatMulMatMul m_24/flatten_15/Reshape:output:0+m_24/dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
m_24/dense_15/MatMul?
$m_24/dense_15/BiasAdd/ReadVariableOpReadVariableOp-m_24_dense_15_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02&
$m_24/dense_15/BiasAdd/ReadVariableOp?
m_24/dense_15/BiasAddBiasAddm_24/dense_15/MatMul:product:0,m_24/dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
m_24/dense_15/BiasAdd?
m_24/dense_15/SigmoidSigmoidm_24/dense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
m_24/dense_15/Sigmoid?
IdentityIdentitym_24/dense_15/Sigmoid:y:0&^m_24/conv2d_25/BiasAdd/ReadVariableOp%^m_24/conv2d_25/Conv2D/ReadVariableOp&^m_24/conv2d_26/BiasAdd/ReadVariableOp%^m_24/conv2d_26/Conv2D/ReadVariableOp%^m_24/dense_15/BiasAdd/ReadVariableOp$^m_24/dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2N
%m_24/conv2d_25/BiasAdd/ReadVariableOp%m_24/conv2d_25/BiasAdd/ReadVariableOp2L
$m_24/conv2d_25/Conv2D/ReadVariableOp$m_24/conv2d_25/Conv2D/ReadVariableOp2N
%m_24/conv2d_26/BiasAdd/ReadVariableOp%m_24/conv2d_26/BiasAdd/ReadVariableOp2L
$m_24/conv2d_26/Conv2D/ReadVariableOp$m_24/conv2d_26/Conv2D/ReadVariableOp2L
$m_24/dense_15/BiasAdd/ReadVariableOp$m_24/dense_15/BiasAdd/ReadVariableOp2J
#m_24/dense_15/MatMul/ReadVariableOp#m_24/dense_15/MatMul/ReadVariableOp:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
?
'__inference_m_24_layer_call_fn_38891239

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_24_layer_call_and_return_conditional_losses_388910872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_15_layer_call_and_return_conditional_losses_38890959

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$@:W S
/
_output_shapes
:?????????$@
 
_user_specified_nameinputs
?"
?
B__inference_m_24_layer_call_and_return_conditional_losses_38891019
conv2d_25_input
conv2d_25_38890998
conv2d_25_38891000
conv2d_26_38891005
conv2d_26_38891007
dense_15_38891013
dense_15_38891015
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputconv2d_25_38890998conv2d_25_38891000*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_388908482#
!conv2d_25/StatefulPartitionedCall?
dropout_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908812
dropout_25/PartitionedCall?
 max_pooling2d_25/PartitionedCallPartitionedCall#dropout_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_388908152"
 max_pooling2d_25/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_38891005conv2d_26_38891007*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_388909062#
!conv2d_26/StatefulPartitionedCall?
dropout_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909392
dropout_26/PartitionedCall?
 max_pooling2d_26/PartitionedCallPartitionedCall#dropout_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_388908272"
 max_pooling2d_26/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_15_layer_call_and_return_conditional_losses_388909592
flatten_15/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_15_38891013dense_15_38891015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_388909782"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
I
-__inference_dropout_25_layer_call_fn_38891286

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908812
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_38890827

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_26_layer_call_fn_38891328

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs
?
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891323

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????I@:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_25_layer_call_fn_38890821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_388908152
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_25_layer_call_and_return_conditional_losses_38890848

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
'__inference_m_24_layer_call_fn_38891102
conv2d_25_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_24_layer_call_and_return_conditional_losses_388910872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
f
H__inference_dropout_25_layer_call_and_return_conditional_losses_38890881

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????? 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????? 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?

?
G__inference_conv2d_26_layer_call_and_return_conditional_losses_38891297

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????I@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????T ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????T 
 
_user_specified_nameinputs
?
?
'__inference_m_24_layer_call_fn_38891061
conv2d_25_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_24_layer_call_and_return_conditional_losses_388910462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
f
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891276

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:?????????? 2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:?????????? 2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
j
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_38890815

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
O
3__inference_max_pooling2d_26_layer_call_fn_38890833

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_388908272
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_m_24_layer_call_fn_38891222

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_m_24_layer_call_and_return_conditional_losses_388910462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?'
?
B__inference_m_24_layer_call_and_return_conditional_losses_38891205

inputs,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity?? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2Dinputs'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingVALID*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
conv2d_25/Relu?
dropout_25/IdentityIdentityconv2d_25/Relu:activations:0*
T0*0
_output_shapes
:?????????? 2
dropout_25/Identity?
max_pooling2d_25/MaxPoolMaxPooldropout_25/Identity:output:0*/
_output_shapes
:?????????T *
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I@2
conv2d_26/Relu?
dropout_26/IdentityIdentityconv2d_26/Relu:activations:0*
T0*/
_output_shapes
:?????????I@2
dropout_26/Identity?
max_pooling2d_26/MaxPoolMaxPooldropout_26/Identity:output:0*/
_output_shapes
:?????????$@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_15/Const?
flatten_15/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_15/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_15/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
dense_15/Sigmoid?
IdentityIdentitydense_15/Sigmoid:y:0!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
I
-__inference_dropout_26_layer_call_fn_38891333

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909392
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I@:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs
?
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891318

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I@:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs
?
?
+__inference_dense_15_layer_call_fn_38891364

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_388909782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
g
H__inference_dropout_26_layer_call_and_return_conditional_losses_38890934

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????I@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????I@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I@2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I@2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????I@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????I@:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs
?:
?
B__inference_m_24_layer_call_and_return_conditional_losses_38891174

inputs,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource-
)conv2d_26_biasadd_readvariableop_resource+
'dense_15_matmul_readvariableop_resource,
(dense_15_biasadd_readvariableop_resource
identity?? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_15/BiasAdd/ReadVariableOp?dense_15/MatMul/ReadVariableOp?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2Dinputs'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingVALID*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2
conv2d_25/BiasAdd
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
conv2d_25/Reluy
dropout_25/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_25/dropout/Const?
dropout_25/dropout/MulMulconv2d_25/Relu:activations:0!dropout_25/dropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout_25/dropout/Mul?
dropout_25/dropout/ShapeShapeconv2d_25/Relu:activations:0*
T0*
_output_shapes
:2
dropout_25/dropout/Shape?
/dropout_25/dropout/random_uniform/RandomUniformRandomUniform!dropout_25/dropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype021
/dropout_25/dropout/random_uniform/RandomUniform?
!dropout_25/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_25/dropout/GreaterEqual/y?
dropout_25/dropout/GreaterEqualGreaterEqual8dropout_25/dropout/random_uniform/RandomUniform:output:0*dropout_25/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2!
dropout_25/dropout/GreaterEqual?
dropout_25/dropout/CastCast#dropout_25/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout_25/dropout/Cast?
dropout_25/dropout/Mul_1Muldropout_25/dropout/Mul:z:0dropout_25/dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout_25/dropout/Mul_1?
max_pooling2d_25/MaxPoolMaxPooldropout_25/dropout/Mul_1:z:0*/
_output_shapes
:?????????T *
ksize
*
paddingVALID*
strides
2
max_pooling2d_25/MaxPool?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2D!max_pooling2d_25/MaxPool:output:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@*
paddingVALID*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????I@2
conv2d_26/BiasAdd~
conv2d_26/ReluReluconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????I@2
conv2d_26/Reluy
dropout_26/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_26/dropout/Const?
dropout_26/dropout/MulMulconv2d_26/Relu:activations:0!dropout_26/dropout/Const:output:0*
T0*/
_output_shapes
:?????????I@2
dropout_26/dropout/Mul?
dropout_26/dropout/ShapeShapeconv2d_26/Relu:activations:0*
T0*
_output_shapes
:2
dropout_26/dropout/Shape?
/dropout_26/dropout/random_uniform/RandomUniformRandomUniform!dropout_26/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????I@*
dtype021
/dropout_26/dropout/random_uniform/RandomUniform?
!dropout_26/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!dropout_26/dropout/GreaterEqual/y?
dropout_26/dropout/GreaterEqualGreaterEqual8dropout_26/dropout/random_uniform/RandomUniform:output:0*dropout_26/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????I@2!
dropout_26/dropout/GreaterEqual?
dropout_26/dropout/CastCast#dropout_26/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????I@2
dropout_26/dropout/Cast?
dropout_26/dropout/Mul_1Muldropout_26/dropout/Mul:z:0dropout_26/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????I@2
dropout_26/dropout/Mul_1?
max_pooling2d_26/MaxPoolMaxPooldropout_26/dropout/Mul_1:z:0*/
_output_shapes
:?????????$@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_26/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten_15/Const?
flatten_15/ReshapeReshape!max_pooling2d_26/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_15/Reshape?
dense_15/MatMul/ReadVariableOpReadVariableOp'dense_15_matmul_readvariableop_resource*
_output_shapes
:	?X*
dtype02 
dense_15/MatMul/ReadVariableOp?
dense_15/MatMulMatMulflatten_15/Reshape:output:0&dense_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_15/MatMul?
dense_15/BiasAdd/ReadVariableOpReadVariableOp(dense_15_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02!
dense_15/BiasAdd/ReadVariableOp?
dense_15/BiasAddBiasAdddense_15/MatMul:product:0'dense_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????X2
dense_15/BiasAdd|
dense_15/SigmoidSigmoiddense_15/BiasAdd:output:0*
T0*'
_output_shapes
:?????????X2
dense_15/Sigmoid?
IdentityIdentitydense_15/Sigmoid:y:0!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^dense_15/BiasAdd/ReadVariableOp^dense_15/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2B
dense_15/BiasAdd/ReadVariableOpdense_15/BiasAdd/ReadVariableOp2@
dense_15/MatMul/ReadVariableOpdense_15/MatMul/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
-__inference_dropout_25_layer_call_fn_38891281

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_38890876

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?!
?
B__inference_m_24_layer_call_and_return_conditional_losses_38891087

inputs
conv2d_25_38891066
conv2d_25_38891068
conv2d_26_38891073
conv2d_26_38891075
dense_15_38891081
dense_15_38891083
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25_38891066conv2d_25_38891068*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_388908482#
!conv2d_25/StatefulPartitionedCall?
dropout_25/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908812
dropout_25/PartitionedCall?
 max_pooling2d_25/PartitionedCallPartitionedCall#dropout_25/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_388908152"
 max_pooling2d_25/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_38891073conv2d_26_38891075*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_388909062#
!conv2d_26/StatefulPartitionedCall?
dropout_26/PartitionedCallPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909392
dropout_26/PartitionedCall?
 max_pooling2d_26/PartitionedCallPartitionedCall#dropout_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_388908272"
 max_pooling2d_26/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_15_layer_call_and_return_conditional_losses_388909592
flatten_15/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_15_38891081dense_15_38891083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_388909782"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_conv2d_25_layer_call_fn_38891259

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_388908482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
G__inference_conv2d_25_layer_call_and_return_conditional_losses_38891250

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? *
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????? 2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
B__inference_m_24_layer_call_and_return_conditional_losses_38890995
conv2d_25_input
conv2d_25_38890859
conv2d_25_38890861
conv2d_26_38890917
conv2d_26_38890919
dense_15_38890989
dense_15_38890991
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?"dropout_25/StatefulPartitionedCall?"dropout_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputconv2d_25_38890859conv2d_25_38890861*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_388908482#
!conv2d_25/StatefulPartitionedCall?
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908762$
"dropout_25/StatefulPartitionedCall?
 max_pooling2d_25/PartitionedCallPartitionedCall+dropout_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_388908152"
 max_pooling2d_25/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_38890917conv2d_26_38890919*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_388909062#
!conv2d_26/StatefulPartitionedCall?
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909342$
"dropout_26/StatefulPartitionedCall?
 max_pooling2d_26/PartitionedCallPartitionedCall+dropout_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_388908272"
 max_pooling2d_26/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_15_layer_call_and_return_conditional_losses_388909592
flatten_15/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_15_38890989dense_15_38890991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_388909782"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
d
H__inference_flatten_15_layer_call_and_return_conditional_losses_38891339

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????$@:W S
/
_output_shapes
:?????????$@
 
_user_specified_nameinputs
?
g
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891271

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:?????????? *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:?????????? 2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:?????????? 2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:?????????? 2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????? :X T
0
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?$
?
B__inference_m_24_layer_call_and_return_conditional_losses_38891046

inputs
conv2d_25_38891025
conv2d_25_38891027
conv2d_26_38891032
conv2d_26_38891034
dense_15_38891040
dense_15_38891042
identity??!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall? dense_15/StatefulPartitionedCall?"dropout_25/StatefulPartitionedCall?"dropout_26/StatefulPartitionedCall?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_25_38891025conv2d_25_38891027*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_388908482#
!conv2d_25/StatefulPartitionedCall?
"dropout_25/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_25_layer_call_and_return_conditional_losses_388908762$
"dropout_25/StatefulPartitionedCall?
 max_pooling2d_25/PartitionedCallPartitionedCall+dropout_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????T * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_388908152"
 max_pooling2d_25/PartitionedCall?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_25/PartitionedCall:output:0conv2d_26_38891032conv2d_26_38891034*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_388909062#
!conv2d_26/StatefulPartitionedCall?
"dropout_26/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:0#^dropout_25/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????I@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_dropout_26_layer_call_and_return_conditional_losses_388909342$
"dropout_26/StatefulPartitionedCall?
 max_pooling2d_26/PartitionedCallPartitionedCall+dropout_26/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????$@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_388908272"
 max_pooling2d_26/PartitionedCall?
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_26/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *Q
fLRJ
H__inference_flatten_15_layer_call_and_return_conditional_losses_388909592
flatten_15/PartitionedCall?
 dense_15/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_15_38891040dense_15_38891042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_dense_15_layer_call_and_return_conditional_losses_388909782"
 dense_15/StatefulPartitionedCall?
IdentityIdentity)dense_15/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall!^dense_15/StatefulPartitionedCall#^dropout_25/StatefulPartitionedCall#^dropout_26/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2D
 dense_15/StatefulPartitionedCall dense_15/StatefulPartitionedCall2H
"dropout_25/StatefulPartitionedCall"dropout_25/StatefulPartitionedCall2H
"dropout_26/StatefulPartitionedCall"dropout_26/StatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?{
?
$__inference__traced_restore_38891571
file_prefix%
!assignvariableop_conv2d_25_kernel%
!assignvariableop_1_conv2d_25_bias'
#assignvariableop_2_conv2d_26_kernel%
!assignvariableop_3_conv2d_26_bias&
"assignvariableop_4_dense_15_kernel$
 assignvariableop_5_dense_15_bias 
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
+assignvariableop_17_adam_conv2d_25_kernel_m-
)assignvariableop_18_adam_conv2d_25_bias_m/
+assignvariableop_19_adam_conv2d_26_kernel_m-
)assignvariableop_20_adam_conv2d_26_bias_m.
*assignvariableop_21_adam_dense_15_kernel_m,
(assignvariableop_22_adam_dense_15_bias_m/
+assignvariableop_23_adam_conv2d_25_kernel_v-
)assignvariableop_24_adam_conv2d_25_bias_v/
+assignvariableop_25_adam_conv2d_26_kernel_v-
)assignvariableop_26_adam_conv2d_26_bias_v.
*assignvariableop_27_adam_dense_15_kernel_v,
(assignvariableop_28_adam_dense_15_bias_v
identity_30??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*O
valueFBDB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesz
x::::::::::::::::::::::::::::::*,
dtypes"
 2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_26_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_26_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_15_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_15_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp+assignvariableop_17_adam_conv2d_25_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp)assignvariableop_18_adam_conv2d_25_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_adam_conv2d_26_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_adam_conv2d_26_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_dense_15_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_dense_15_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_25_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_25_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_conv2d_26_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_conv2d_26_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_15_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_15_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_289
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_29Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_29?
Identity_30IdentityIdentity_29:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_30"#
identity_30Identity_30:output:0*?
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
?
?
&__inference_signature_wrapper_38891129
conv2d_25_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_25_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_388908092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:??????????::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:??????????
)
_user_specified_nameconv2d_25_input
?
f
H__inference_dropout_26_layer_call_and_return_conditional_losses_38890939

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????I@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????I@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:?????????I@:W S
/
_output_shapes
:?????????I@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
T
conv2d_25_inputA
!serving_default_conv2d_25_input:0??????????<
dense_150
StatefulPartitionedCall:0?????????Xtensorflow/serving/predict:ӄ
?=
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
	variables
trainable_variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?:
_tf_keras_sequential?9{"class_name": "Sequential", "name": "m_24", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "m_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_25_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "m_24", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_25_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0006000000284984708, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_25", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_25", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_25", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_25", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?	

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 3, 32]}}
?
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dropout", "name": "dropout_26", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_26", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
?
'regularization_losses
(	variables
)trainable_variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_26", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
+regularization_losses
,	variables
-trainable_variables
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_15", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_15", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?
5iter

6beta_1

7beta_2
	8decay
9learning_ratemxmymzm{/m|0m}v~vv?v?/v?0v?"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
?
:layer_metrics

regularization_losses

;layers
	variables
<non_trainable_variables
=layer_regularization_losses
>metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
*:( 2conv2d_25/kernel
: 2conv2d_25/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
?layer_metrics
regularization_losses

@layers
	variables
Anon_trainable_variables
Blayer_regularization_losses
Cmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_metrics
regularization_losses

Elayers
	variables
Fnon_trainable_variables
Glayer_regularization_losses
Hmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ilayer_metrics
regularization_losses

Jlayers
	variables
Knon_trainable_variables
Llayer_regularization_losses
Mmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_26/kernel
:@2conv2d_26/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Nlayer_metrics
regularization_losses

Olayers
 	variables
Pnon_trainable_variables
Qlayer_regularization_losses
Rmetrics
!trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Slayer_metrics
#regularization_losses

Tlayers
$	variables
Unon_trainable_variables
Vlayer_regularization_losses
Wmetrics
%trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xlayer_metrics
'regularization_losses

Ylayers
(	variables
Znon_trainable_variables
[layer_regularization_losses
\metrics
)trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]layer_metrics
+regularization_losses

^layers
,	variables
_non_trainable_variables
`layer_regularization_losses
ametrics
-trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 	?X2dense_15/kernel
:X2dense_15/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
blayer_metrics
1regularization_losses

clayers
2	variables
dnon_trainable_variables
elayer_regularization_losses
fmetrics
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
g0
h1
i2"
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
?
	jtotal
	kcount
l	variables
m	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
n
thresholds
otrue_positives
pfalse_positives
q	variables
r	keras_api"?
_tf_keras_metric?{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
?
s
thresholds
ttrue_positives
ufalse_negatives
v	variables
w	keras_api"?
_tf_keras_metric?{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
:  (2total
:  (2count
.
j0
k1"
trackable_list_wrapper
-
l	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
t0
u1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
/:- 2Adam/conv2d_25/kernel/m
!: 2Adam/conv2d_25/bias/m
/:- @2Adam/conv2d_26/kernel/m
!:@2Adam/conv2d_26/bias/m
':%	?X2Adam/dense_15/kernel/m
 :X2Adam/dense_15/bias/m
/:- 2Adam/conv2d_25/kernel/v
!: 2Adam/conv2d_25/bias/v
/:- @2Adam/conv2d_26/kernel/v
!:@2Adam/conv2d_26/bias/v
':%	?X2Adam/dense_15/kernel/v
 :X2Adam/dense_15/bias/v
?2?
'__inference_m_24_layer_call_fn_38891222
'__inference_m_24_layer_call_fn_38891102
'__inference_m_24_layer_call_fn_38891239
'__inference_m_24_layer_call_fn_38891061?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_38890809?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/
conv2d_25_input??????????
?2?
B__inference_m_24_layer_call_and_return_conditional_losses_38891174
B__inference_m_24_layer_call_and_return_conditional_losses_38891019
B__inference_m_24_layer_call_and_return_conditional_losses_38891205
B__inference_m_24_layer_call_and_return_conditional_losses_38890995?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_conv2d_25_layer_call_fn_38891259?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_25_layer_call_and_return_conditional_losses_38891250?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dropout_25_layer_call_fn_38891286
-__inference_dropout_25_layer_call_fn_38891281?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891271
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891276?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_max_pooling2d_25_layer_call_fn_38890821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_38890815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_conv2d_26_layer_call_fn_38891306?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_conv2d_26_layer_call_and_return_conditional_losses_38891297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_dropout_26_layer_call_fn_38891333
-__inference_dropout_26_layer_call_fn_38891328?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891323
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891318?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_max_pooling2d_26_layer_call_fn_38890833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_38890827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
-__inference_flatten_15_layer_call_fn_38891344?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_flatten_15_layer_call_and_return_conditional_losses_38891339?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_dense_15_layer_call_fn_38891364?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dense_15_layer_call_and_return_conditional_losses_38891355?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_38891129conv2d_25_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_38890809?/0A?>
7?4
2?/
conv2d_25_input??????????
? "3?0
.
dense_15"?
dense_15?????????X?
G__inference_conv2d_25_layer_call_and_return_conditional_losses_38891250n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????? 
? ?
,__inference_conv2d_25_layer_call_fn_38891259a8?5
.?+
)?&
inputs??????????
? "!??????????? ?
G__inference_conv2d_26_layer_call_and_return_conditional_losses_38891297l7?4
-?*
(?%
inputs?????????T 
? "-?*
#? 
0?????????I@
? ?
,__inference_conv2d_26_layer_call_fn_38891306_7?4
-?*
(?%
inputs?????????T 
? " ??????????I@?
F__inference_dense_15_layer_call_and_return_conditional_losses_38891355]/00?-
&?#
!?
inputs??????????
? "%?"
?
0?????????X
? 
+__inference_dense_15_layer_call_fn_38891364P/00?-
&?#
!?
inputs??????????
? "??????????X?
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891271n<?9
2?/
)?&
inputs?????????? 
p
? ".?+
$?!
0?????????? 
? ?
H__inference_dropout_25_layer_call_and_return_conditional_losses_38891276n<?9
2?/
)?&
inputs?????????? 
p 
? ".?+
$?!
0?????????? 
? ?
-__inference_dropout_25_layer_call_fn_38891281a<?9
2?/
)?&
inputs?????????? 
p
? "!??????????? ?
-__inference_dropout_25_layer_call_fn_38891286a<?9
2?/
)?&
inputs?????????? 
p 
? "!??????????? ?
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891318l;?8
1?.
(?%
inputs?????????I@
p
? "-?*
#? 
0?????????I@
? ?
H__inference_dropout_26_layer_call_and_return_conditional_losses_38891323l;?8
1?.
(?%
inputs?????????I@
p 
? "-?*
#? 
0?????????I@
? ?
-__inference_dropout_26_layer_call_fn_38891328_;?8
1?.
(?%
inputs?????????I@
p
? " ??????????I@?
-__inference_dropout_26_layer_call_fn_38891333_;?8
1?.
(?%
inputs?????????I@
p 
? " ??????????I@?
H__inference_flatten_15_layer_call_and_return_conditional_losses_38891339a7?4
-?*
(?%
inputs?????????$@
? "&?#
?
0??????????
? ?
-__inference_flatten_15_layer_call_fn_38891344T7?4
-?*
(?%
inputs?????????$@
? "????????????
B__inference_m_24_layer_call_and_return_conditional_losses_38890995z/0I?F
??<
2?/
conv2d_25_input??????????
p

 
? "%?"
?
0?????????X
? ?
B__inference_m_24_layer_call_and_return_conditional_losses_38891019z/0I?F
??<
2?/
conv2d_25_input??????????
p 

 
? "%?"
?
0?????????X
? ?
B__inference_m_24_layer_call_and_return_conditional_losses_38891174q/0@?=
6?3
)?&
inputs??????????
p

 
? "%?"
?
0?????????X
? ?
B__inference_m_24_layer_call_and_return_conditional_losses_38891205q/0@?=
6?3
)?&
inputs??????????
p 

 
? "%?"
?
0?????????X
? ?
'__inference_m_24_layer_call_fn_38891061m/0I?F
??<
2?/
conv2d_25_input??????????
p

 
? "??????????X?
'__inference_m_24_layer_call_fn_38891102m/0I?F
??<
2?/
conv2d_25_input??????????
p 

 
? "??????????X?
'__inference_m_24_layer_call_fn_38891222d/0@?=
6?3
)?&
inputs??????????
p

 
? "??????????X?
'__inference_m_24_layer_call_fn_38891239d/0@?=
6?3
)?&
inputs??????????
p 

 
? "??????????X?
N__inference_max_pooling2d_25_layer_call_and_return_conditional_losses_38890815?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_25_layer_call_fn_38890821?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
N__inference_max_pooling2d_26_layer_call_and_return_conditional_losses_38890827?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_max_pooling2d_26_layer_call_fn_38890833?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
&__inference_signature_wrapper_38891129?/0T?Q
? 
J?G
E
conv2d_25_input2?/
conv2d_25_input??????????"3?0
.
dense_15"?
dense_15?????????X