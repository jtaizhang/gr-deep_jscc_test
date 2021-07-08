/* -*- c++ -*- */
/*
 * Copyright 2021 jtaizhang.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_DEEP_JSCC_TEST_OFDM_JSCC_EQUALIZER_SINGLE_TAP_H
#define INCLUDED_DEEP_JSCC_TEST_OFDM_JSCC_EQUALIZER_SINGLE_TAP_H

#include <deep_jscc_test/api.h>
#include <gnuradio/digital/ofdm_equalizer_base.h>
#include <iostream>

namespace gr {
  namespace deep_jscc_test {

    /*!
     * \brief <+description+>
     *
     */
    class DEEP_JSCC_TEST_API ofdm_jscc_equalizer_single_tap : public gr::digital::ofdm_equalizer_1d_pilots
    {
    public:
      typedef boost::shared_ptr<ofdm_jscc_equalizer_single_tap> sptr;
      ofdm_jscc_equalizer_single_tap(
        int fft_len,
        const std::vector<std::vector<int>> &occupied_carriers,
        const std::vector<std::vector<int>> &pilot_carriers,
        const std::vector<std::vector<gr_complex>> &pilot_symbols,
        int symbols_skipped,
        bool input_is_shifted);

      void equalize(gr_complex *frame,
                    int n_sym,
                    const std::vector<gr_complex> &initial_taps = std::vector<gr_complex>(),
                    const std::vector<tag_t> &tags = std::vector<tag_t>());

      static sptr make(
        int fft_len,
        const std::vector<std::vector<int>> &occupied_carriers,
        const std::vector<std::vector<int>> &pilot_carriers,
        const std::vector<std::vector<gr_complex>> &pilot_symbols,
        int symbols_skipped = 0,
        bool input_is_shifted = true);
      // ~ofdm_jscc_equalizer_single_tap();
    private:
    };

  } // namespace deep_jscc_test
} // namespace gr

#endif /* INCLUDED_DEEP_JSCC_TEST_OFDM_JSCC_EQUALIZER_SINGLE_TAP_H */

