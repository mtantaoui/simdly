// #[inline(always)]
// fn pack_panel_a_into(
//     dest_slice: &mut [f32],
//     a_panel_source_slice: &[f32],
//     mr_effective_in_panel: usize,
//     kc_panel: usize,
//     m_original_matrix: usize,
// ) {
//     debug_assert_eq!(
//         dest_slice.len(),
//         kc_panel * MR,
//         "Destination slice length mismatch."
//     );
//     debug_assert!(
//         mr_effective_in_panel <= MR,
//         "mr_effective_in_panel cannot exceed MR."
//     );
//     debug_assert!(
//         MR == f32x8::LANE_COUNT,
//         "MR must match F32x8::LANE_COUNT for this SIMD pack_panel_a_into version"
//     );

//     for p_col_in_panel in 0..kc_panel {
//         // Iterate over columns of the panel (K-dimension)
//         let source_col_start_offset = p_col_in_panel * m_original_matrix;
//         let dest_col_segment_start_offset = p_col_in_panel * MR;

//         let dest_ptr = unsafe { dest_slice.as_mut_ptr().add(dest_col_segment_start_offset) };

//         // The source elements for one column of the A_panel are contiguous.
//         // Slice the source for the current column of A's panel.
//         // It has `mr_effective_in_panel` actual data elements.
//         let current_a_col_slice = unsafe {
//             std::slice::from_raw_parts(
//                 a_panel_source_slice.as_ptr().add(source_col_start_offset),
//                 mr_effective_in_panel,
//             )
//         };

//         // F32x8::new will call load_partial if mr_effective_in_panel < MR (i.e. < LANE_COUNT),
//         // zero-padding the SIMD vector. If mr_effective_in_panel == MR, it calls load.
//         let a_simd_col = F32x8::new(current_a_col_slice);
//         // a_simd_col.elements now contains the mr_effective_in_panel data items,
//         // followed by zeros if mr_effective_in_panel < MR.
//         // a_simd_col.size is mr_effective_in_panel.

//         unsafe {
//             // Store the full MR-element (LANE_COUNT-element) SIMD vector.
//             // This writes the data and any necessary zero-padding within this MR-block.
//             // This is correct because the destination segment is exactly MR elements long.
//             // F32x8::store_at will handle aligned/unaligned based on dest_ptr.
//             // Since dest_slice comes from an aligned allocation and MR * sizeof(f32) is
//             // a multiple of alignment, dest_ptr will be aligned.
//             a_simd_col.store_at(dest_ptr);
//         }
//     }
//     // The explicit scalar padding loop is no longer needed because F32x8::new + F32x8::store_at
//     // handles the padding within each MR-sized block.
// }

// #[inline(always)]
// fn pack_panel_b(
//     dest_slice: &mut [f32],
//     b_panel_source_slice: &[f32], // Renamed from b_panel for clarity
//     nr_effective_in_panel: usize, // Renamed from nr
//     kc_panel: usize,              // Renamed from kc
//     k_original_matrix: usize,     // Renamed from k
// ) {
//     debug_assert_eq!(
//         dest_slice.len(),
//         kc_panel * NR,
//         "Destination slice length incorrect. Expected {}, got {}. kc_panel={}, NR={}",
//         kc_panel * NR,
//         dest_slice.len(),
//         kc_panel,
//         NR
//     );
//     debug_assert!(
//         nr_effective_in_panel <= NR,
//         "nr_effective_in_panel ({nr_effective_in_panel}) cannot exceed NR ({NR})"
//     );
//     // NR < F32x8::LANE_COUNT is implicitly assumed by the store_at_partial strategy if NR were LANE_COUNT
//     // but NR=6, LANE_COUNT=8, so this is fine.

//     // Temporary array to hold one row of the B panel (NR elements).
//     // Initialized to zero to handle padding if nr_effective_in_panel < NR.
//     let mut temp_row_for_simd: [f32; NR] = [0.0; NR]; // NR=6 in this case

//     for p_row_in_panel in 0..kc_panel {
//         let dest_row_start_offset = p_row_in_panel * NR;
//         let dest_ptr = unsafe { dest_slice.as_mut_ptr().add(dest_row_start_offset) };

//         // Scalar gather: Copy nr_effective_in_panel elements from strided b_panel_source_slice
//         // into the start of temp_row_for_simd. The rest of temp_row_for_simd remains zero (padding).
//         if nr_effective_in_panel > 0 {
//             // Optimization: avoid loop if no elements
//             for j_col_in_panel in 0..nr_effective_in_panel {
//                 let source_index = j_col_in_panel * k_original_matrix + p_row_in_panel;
//                 temp_row_for_simd[j_col_in_panel] = b_panel_source_slice[source_index];
//             }
//             // If nr_effective_in_panel < NR, zero out the rest of temp_row_for_simd explicitly
//             // (already done by initialization, but good to be clear if initialization changes)
//             if nr_effective_in_panel < NR {
//                 for j_pad_col in nr_effective_in_panel..NR {
//                     temp_row_for_simd[j_pad_col] = 0.0;
//                 }
//             }
//         } else {
//             // if nr_effective_in_panel is 0, ensure temp_row is all zeros
//             for i in 0..NR {
//                 temp_row_for_simd[i] = 0.0;
//             }
//         }

//         // Now, temp_row_for_simd contains the NR elements (data + padding) for the current row of B's panel.
//         // Create a slice of NR elements to pass to F32x8::new.
//         let current_b_row_slice = &temp_row_for_simd[0..NR];

//         // F32x8::new will call load_partial because NR (6) < LANE_COUNT (8).
//         // b_simd_row.elements will have NR data values from temp_row_for_simd,
//         // followed by (LANE_COUNT - NR) zeros.
//         // b_simd_row.size will be NR.
//         let b_simd_row = F32x8::new(current_b_row_slice);

//         unsafe {
//             // Store exactly NR elements from b_simd_row.elements into dest_slice.
//             // F32x8::store_at_partial uses b_simd_row.size (which is NR) to create
//             // a mask, writing only the first NR elements.
//             // This is correct as the destination segment is NR wide.
//             // _mm256_maskstore_ps does not require dest_ptr to be aligned.
//             b_simd_row.store_at_partial(dest_ptr);
//         }
//     }
//     // The explicit scalar padding loop for dest_slice (nr_effective_in_panel..NR)
//     // is handled by the temp_row_for_simd being correctly prepared with zeros
//     // and then store_at_partial writing these NR values.
// }

fn main() {}
