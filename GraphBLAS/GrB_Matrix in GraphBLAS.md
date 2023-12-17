### GrB_Matrix

#### data structure

```c
typedef struct GB_Matrix_opaque *GrB_Matrix ;
```

```c
struct GB_Matrix_opaque     // content of GrB_Matrix
{
	GrB_Type type ;         // the type of each numerical entry
    bool is_csc ;           // true if stored by column, false if by row
    bool jumbled ;          // true if the matrix may be jumbled.  bitmap and full
                            // matrices are never jumbled.
    int64_t *h ;            // list of non-empty vectors: h_size >= 8*max(plen,1)
    int64_t *p ;            // pointers: p_size >= 8*(plen+1)
    int64_t *i ;            // indices:  i_size >= 8*max(anz,1)
    void *x ;               // values:   x_size >= max(anz*A->type->size,1),
                            //           or x_size >= 1 if A is iso
    int8_t *b ;             // bitmap:   b_size >= max(anz,1)
    int64_t nvals ;         // nvals(A) if A is bitmap

    size_t p_size ;         // exact size of A->p in bytes, zero if A->p is NULL
    size_t h_size ;         // exact size of A->h in bytes, zero if A->h is NULL
    size_t b_size ;         // exact size of A->b in bytes, zero if A->b is NULL
    size_t i_size ;         // exact size of A->i in bytes, zero if A->i is NULL
    size_t x_size ;         // exact size of A->x in bytes, zero if A->x is NULL

    GB_Pending Pending ;        // list of pending tuples
} ;
```

```c
struct GB_Pending_struct    // list of pending tuples for a matrix
{
    size_t header_size ;    // size of the malloc'd block for this struct, or 0
    int64_t n ;         // number of pending tuples to add to matrix
    int64_t nmax ;      // size of i,j,x
    bool sorted ;       // true if pending tuples are in sorted order
    int64_t *i ;        // row indices of pending tuples
    size_t i_size ;
    int64_t *j ;        // col indices of pending tuples; NULL if A->vdim <= 1
    size_t j_size ;
    GB_void *x ;        // values of pending tuples
    size_t x_size ;
    GrB_Type type ;     // the type of s
    size_t size ;       // type->size
    GrB_BinaryOp op ;   // operator to assemble pending tuples
} ;

typedef struct GB_Pending_struct *GB_Pending ;
```

- Sparse structure:

  ```c
  // Row A(i,:) is held in two parts: the column indices are in
  // Ai [Ap [i]...Ap [i+1]-1], and the numerical values are in the
  ```

- Hypersparse structure:

  ```c
  // If row A(i,:) has any entries, then i = Ah [k] for some
  // k in the range 0 to A->nvec-1.
  
  // Row A(i,:) is held in two parts: the column indices are in Ai
  // [Ap [k]...Ap [k+1]-1], and the numerical values are in the same positions in Ax.
  ```

#### api

##### GrB_Matrix_new

```c
GrB_Info GrB_Matrix_new     // create a new matrix with no entries
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // matrix dimension is nrows-by-ncols
    GrB_Index ncols         // (nrows and ncols must be <= GrB_INDEX_MAX+1)
) ;
```

如果ncols=1，则矩阵为csc矩阵

如果nrows=1，则矩阵为csr矩阵

否则，矩阵为默认的类型，csr

```c
int64_t GB_nnz      // return nnz(A) or INT64_MAX if integer overflow
(
    GrB_Matrix A
)
{

    if (A == NULL || A->magic != GB_MAGIC || A->x == NULL)
    { 
        // A is NULL or uninitialized
        return (0) ;
    }
    else if (A->p != NULL)
    { 
        // A is sparse or hypersparse
        return (A->p [A->nvec]) ;
    }
    else if (A->b != NULL)
    { 
        // A is bitmap
        return (A->nvals) ;
    }
    else
    { 
        // A is full
        return (GB_nnz_full (A)) ;
    }
}
```

